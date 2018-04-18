# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for allreduce."""

from __future__ import print_function

import collections as pycoll
import re
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib import nccl
from tensorflow.contrib.all_reduce.python import all_reduce
from allreduce import *


def sum_gradients_all_reduce(dev_prefixes,
                             tower_grads,
                             num_workers,
                             alg,
                             num_shards,
                             gpu_indices,
                             agg_small_grads_max_bytes=0,
                             agg_small_grads_max_group=10):
  """Apply all-reduce algorithm over specified gradient tensors.

  Args:
    dev_prefixes: list of prefix strings to use to generate PS device names.
    tower_grads: the gradients to reduce.
    num_workers: number of worker processes across entire job.
    alg: the all-reduce algorithm to apply.
    num_shards: alg-specific sharding factor.
    gpu_indices: indices of local GPUs in order usable for ring-reduce.
    agg_small_grads_max_bytes: largest tensor eligible for aggregation,
      in number of bytes.
    agg_small_grads_max_group: largest permitted aggregation of small
      tensors.

  Returns:
    list of reduced tensors, packing values
  """
  alg_contains_shuffle = contains_any(alg, ['pscpu', 'psgpu'])
  is_hierarchical = '/' in alg
  if 'pscpu' in alg:
    aux_devices = [prefix + '/cpu:0' for prefix in dev_prefixes]
  elif 'psgpu' in alg:
    aux_devices = [
        prefix + '/gpu:%d' % i
        for i in range(len(gpu_indices))
        for prefix in dev_prefixes
    ]
  else:
    aux_devices = ['/job:localhost/cpu:0']
  aux_device_groups = group_device_names(aux_devices, num_shards
                                         if alg_contains_shuffle else 1)
  group_index = 0
  if agg_small_grads_max_bytes > 0 and agg_small_grads_max_group > 0:
    tower_grads, packing = pack_small_tensors(
        tower_grads,
        max_bytes=agg_small_grads_max_bytes,
        max_group=agg_small_grads_max_group)
  else:
    packing = None
  reduced_gv_list = []
  for grad_and_vars in zip(*tower_grads):
    reduced_gv_list.append(
        sum_grad_and_var_all_reduce(
            grad_and_vars, num_workers, alg, gpu_indices, aux_devices
            if is_hierarchical else aux_device_groups[group_index], num_shards))
    group_index = (group_index + 1) % len(aux_device_groups)
  new_tower_grads = [list(x) for x in zip(*reduced_gv_list)]
  return new_tower_grads, packing



def pack_small_tensors(tower_grads, max_bytes=0):
  """Concatenate gradients together in a more intelligent manner.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples.
    max_bytes: Int giving max number of bytes in a tensor that
      may be considered small.
    max_group: Int giving max number of small tensors that may be
      concatenated into one new tensor.
  """
  assert max_bytes >= 0
  orig_grads = [g for g, _ in tower_grads[0]]
  # Check to make sure sizes are accurate; not entirely important
  assert all(g.dtype == tf.float32 for g in orig_grads)
  sizes = [4 * g.shape.num_elements() for g in orig_grads]
  print("Before packing - avg size %f, median size %f" % (
    np.mean(sizes), np.median(sizes)))
  small_ranges = []
  large_indices = []

  def end_interval(indices):
    if len(indices) > 1:
      small_ranges += [[indices[0], indices[-1]]]
    else:
      large_indices += indices

  cur_range = []
  cur_size = 0
  for i, s in enumerate(sizes):
    if cur_size > max_bytes:
      end_interval(cur_range)
      cur_range = []
      cur_size = 0
    cur_range += [i]
    cur_size += s
  end_interval(cur_range)

  if len(small_ranges):
    new_tower_grads = []
    for dev_idx, gv_list in enumerate(tower_grads):
      assert len(gv_list) == num_gv
      new_gv_list = []
      for r in small_ranges:
        key = '%d:%d' % (dev_idx, len(new_gv_list))
        new_gv_list.append((pack_range(key, packing, gv_list, r),
                            'packing_var_placeholder'))
      for i in large_indices:
        new_gv_list.append(gv_list[i])
      new_tower_grads.append(new_gv_list)
    return new_tower_grads, packing
  else:
    return tower_grads, None
