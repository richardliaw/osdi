#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tfbench import model_config, allreduce
import os
import ray
import time


class MockDataset():
    name = "synthetic"


class TFBenchModel(object):
    def __init__(self, batch=64):
        image_shape = [batch, 224, 224, 3]
        labels_shape = [batch]

        # Synthetic image should be within [0, 255].
        images = tf.truncated_normal(
            image_shape,
            dtype=tf.float32,
            mean=127,
            stddev=60,
            name='synthetic_images')

        # Minor hack to avoid H2D copy when using synthetic data
        self.inputs = tf.contrib.framework.local_variable(
            images, name='gpu_cached_images')
        self.labels = tf.random_uniform(
            labels_shape,
            minval=0,
            maxval=999,
            dtype=tf.int32,
            name='synthetic_labels')

        self.model = model_config.get_model_config("resnet101", MockDataset())
        logits, aux = self.model.build_network(self.inputs, data_format="NCHW")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='xentropy-loss')
        self.optimizer = tf.train.GradientDescentOptimizer(1e-6)


class SGDWorker(object):
    def __init__(self,
                 i,
                 model_cls,
                 batch_size,
                 all_reduce_alg=None,
                 num_devices=1,
                 use_cpus=False,
                 max_bytes=0,
                 plasma_op=False):
        # TODO - just port VariableMgrLocalReplicated
        self.i = i
        assert num_devices > 0
        tf_session_args = {
            "device_count": {"CPU": num_devices}
        }
        config_proto = tf.ConfigProto(**tf_session_args)
        self.sess = tf.Session(config=config_proto)
        models = []
        grad_ops = []

        if use_cpus:
            device_tmpl = "/cpu:%d"
        else:
            device_tmpl = "/gpu:%d"
        for device_idx in range(num_devices):
            with tf.device(device_tmpl % device_idx):
                with tf.variable_scope("device_%d" % device_idx):
                    model = model_cls(batch=batch_size)
                    models += [model]
                    model.grads = [
                        t for t in model.optimizer.compute_gradients(model.loss)
                        if t[0] is not None]
                    grad_ops.append(model.grads)

        self.models = models
        if num_devices == 1:
           self.per_device_grads_and_vars = grad_ops
        elif all_reduce_alg:
            if max_bytes:
                from tfbench import modified_allreduce
                self.per_device_grads_and_vars, packing_vals = modified_allreduce.sum_gradients_all_reduce(
                    "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)),
                    agg_small_grads_max_bytes=max_bytes,
                    agg_small_grads_max_group=9999)
            else:
                self.per_device_grads_and_vars = allreduce.sum_gradients_all_reduce(
                    "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)))
                assert(len(self.per_device_grads_and_vars) == num_devices)
                assert(len(self.per_device_grads_and_vars[0]) == 314)
                assert(len(self.per_device_grads_and_vars[0][0]) == 2)
        self.per_device_grads = [list(zip(*dev_gv))[0] for dev_gv in self.per_device_grads_and_vars]
        assert(len(self.per_device_grads) == num_devices)
        assert(len(self.per_device_grads_and_vars[0]) == 314)

        if plasma_op:
            memcpy_plasma_module = tf.load_op_library("../ops/memcpy_plasma_op.so")

            # For applying grads <- plasma
            unpacked_gv = []
            self.plasma_out_grads_oids = [
                tf.placeholder(shape=[], dtype=tf.string) for _ in range(314)]
            for i in range(num_devices):
                per_device = []
                for j, (_, v) in enumerate(self.per_device_grads_and_vars[i]):
                    grad_ph = memcpy_plasma_module.plasma_to_tensor(
                        self.plasma_out_grads_oids[j],
                        plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                        plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name)
                    grad_ph = tf.reshape(grad_ph, v.shape)
                    per_device.append((grad_ph, v))
                unpacked_gv.append(per_device)

            # For fetching grads -> plasma
            self.plasma_in_grads = []
            self.plasma_in_grads_oids = [
                tf.placeholder(shape=[], dtype=tf.string) for _ in range(314)]
            all_grads = []
            for per_device in self.per_device_grads:
                all_grads.extend(per_device)
            with tf.control_dependencies(all_grads):
                for j, grad in enumerate(self.per_device_grads[0]):  # from 0th device
                    plasma_grad = memcpy_plasma_module.tensor_to_plasma(
                        grad,
                        self.plasma_in_device_grads_oids[j],
                        plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                        plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name)
                    self.plasma_in_grads.append(plasma_grad)

        elif max_bytes:
            unpacked_gv = allreduce.unpack_small_tensors(self.per_device_grads_and_vars, packing_vals)
        else:
            unpacked_gv = self.per_device_grads_and_vars

        # Same shape as per_device_grads_and_vars
        assert len(unpacked_gv) == num_devices
        assert len(unpacked_gv[0]) == 314
        assert len(unpacked_gv[0][0]) == 2

        self.apply_op = tf.group(
            *[m.optimizer.apply_gradients(g) for g, m in zip(unpacked_gv, models)])
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def compute_apply(self, write_timeline):
       if write_timeline:
           run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
           run_metadata = tf.RunMetadata()
           self.sess.run(self.apply_op,
               options=run_options,
               run_metadata=run_metadata)
           trace = timeline.Timeline(step_stats=run_metadata.step_stats)
           outf = "timeline-sgd.json"
           trace_file = open(outf, "w")
           print("wrote tf timeline to", os.path.abspath(outf))
           trace_file.write(trace.generate_chrome_trace_format())
       else:
          self.sess.run(self.apply_op)

    def compute_gradients(self, verbose):
        start = time.time()
        fetches = self.sess.run(self.per_device_grads)
        if verbose:
            print("compute grad interior time", time.time() - start)
        return fetches[0]

    def apply_gradients(self, avg_grads, verbose):
        start = time.time()
        result = {}
        for per_device_grads_and_vars in self.per_device_grads_and_vars:
            m = {per_device_grads_and_vars[j][0]: grad for j, grad in enumerate(avg_grads)}
            result.update(m)
        self.sess.run(self.apply_op, feed_dict=result)
        if verbose:
            print("apply grad interior time", time.time() - start)

    def compute_gradients_to_plasma_direct(self, verbose):
        plasma_in_grads_oids = [
            np.random.bytes(20) for _ in self.plasma_in_grads_oids]
        start = time.time()
        fetches = self.sess.run(self.plasma_in_grads, feed_dict={
            ph: oid for (ph, oid) in zip(self.plasma_in_grads_oids, plasma_in_grads_oids)
        })
        if verbose:
            print("compute grad plasma interior time", time.time() - start)
        return plasma_in_grads_oids

    def apply_gradients_from_plasma_direct(self, avg_grads_oids, verbose):
        start = time.time()
        feed = {
            ph: oid for (ph, oid) in zip(self.plasma_out_grads_oids, avg_grads_oids)
        }
        self.sess.run(self.apply_op, feed_dict=feed)
        if verbose:
            print("apply grad plasma interior time", time.time() - start)


def average_gradients(grads):
    out = []
    for grad_list in zip(*grads):
        out.append(np.mean(grad_list, axis=0))
    return out


def do_sgd_step(actors, local_only, write_timeline, verbose, plasma_op):
    if local_only:
        ray.get([a.compute_apply.remote(write_timeline) for a in actors])
    else:
        start = time.time()
        if plasma_op:
            grads = ray.get([a.compute_gradients_plasma_direct.remote(verbose) for a in actors])
        else:
            grads = ray.get([a.compute_gradients.remote(verbose) for a in actors])
        if verbose:
            print("compute all grads time", time.time() - start)
        start = time.time()
        if len(actors) == 1:
            assert len(grads) == 1
            avg_grad = grads[0]
        else:
            # TODO(ekl) replace with allreduce
            avg_grad = average_gradients(grads)
        if verbose:
            print("distributed allreduce time", time.time() - start)
        start = time.time()
        if plasma_op:
            ray.get([a.apply_gradients_plasma_direct.remote(avg_grad, verbose) for a in actors])
        else:
            ray.get([a.apply_gradients.remote(avg_grad, verbose) for a in actors])
        if verbose:
            print("apply all grads time", time.time() - start)


import argparse

parser = argparse.ArgumentParser()

# Scaling
parser.add_argument("--devices-per-actor", type=int, default=1,
    help="Number of GPU/CPU towers to use per actor")
parser.add_argument("--num-actors", type=int, default=1,
    help="Number of actors to use for distributed sgd")

# Debug
parser.add_argument("--timeline", action="store_true",
    help="Whether to write out a TF timeline")
parser.add_argument("--verbose", action="store_true",
    help="Whether to print out timing debug messages")
parser.add_argument("--warmup", action="store_true",
    help="Whether to warmup the object store first")
parser.add_argument("--hugepages", action="store_true",
    help="Whether to use hugepages")
parser.add_argument("--local-only", action="store_true",
    help="Whether to skip the object store for performance testing.")
parser.add_argument("--plasma-op", action="store_true",
    help="Whether to use the plasma TF op.")
parser.add_argument("--use-cpus", action="store_true",
    help="Whether to use CPU devices instead of GPU for debugging.")
parser.add_argument("--max-bytes", type=int, default=0,
    help="Max byte tensor to pack")
parser.add_argument("--batch-size", type=int, default=64,
    help="ResNet101 batch size")
parser.add_argument("--allreduce-spec", type=str, default="",
    help="Allreduce spec")


def warmup():
    print("Warming up object store")
    zeros = np.zeros(int(100e6 / 8), dtype=np.float64)
    start = time.time()
    for _ in range(10):
        ray.put(zeros)
    print("Initial latency for 100MB put", (time.time() - start) / 10)
    for _ in range(10):
        for _ in range(100):
            ray.put(zeros)
        start = time.time()
        for _ in range(10):
            ray.put(zeros)
        print("Warming up latency for 100MB put", (time.time() - start) / 10)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.hugepages:
        ray.init(huge_pages=True, plasma_directory="/mnt/hugepages/")
    else:
        ray.init()
    if args.warmup:
        warmup()
    if args.timeline:
        assert args.local_only
    model = TFBenchModel
    if args.use_cpus:
        requests = {"num_cpus": args.devices_per_actor}
    else:
        requests = {"num_gpus": args.devices_per_actor}
    RemoteSGDWorker = ray.remote(**requests)(SGDWorker)
    if args.use_cpus:
        spec = "xring"
    else:
        spec = "nccl"
    if args.allreduce_spec:
        spec = args.allreduce_spec
    actors = [
        RemoteSGDWorker.remote(
            i, model, args.batch_size, spec,
            use_cpus=args.use_cpus, num_devices=args.devices_per_actor, 
            max_bytes=args.max_bytes, plasma_op=args.plasma_op)
        for i in range(args.num_actors)]
    print("Test config: " + str(args))
    for i in range(10):
        start = time.time()
        print("Distributed sgd step", i)
        do_sgd_step(actors, args.local_only, args.timeline, args.verbose, args.plasma_op)
        print("Images per second", args.batch_size * args.num_actors * args.devices_per_actor / (time.time() - start))
