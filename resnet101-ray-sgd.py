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
        logits, aux = self.model.build_network(self.inputs, data_format="NHWC")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='xentropy-loss')
        self.optimizer = tf.train.GradientDescentOptimizer(1e-6)

    def feed_dict(self):
        return {
            self.inputs: np.zeros(self.inputs.shape.as_list()),
            self.labels: np.zeros(self.labels.shape.as_list())
        }


class SGDWorker(object):
    def __init__(self, i, model_cls, batch_size, all_reduce_alg=None, num_devices=1, use_cpus=False):
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
                    model.grads = [t for t in model.optimizer.compute_gradients(model.loss) if t[0] is not None]
                    grad_ops.append(model.grads)

        self.models = models
        if num_devices == 1:
           self.device_grads_and_vars = grad_ops
        elif all_reduce_alg:
           self.device_grads_and_vars = allreduce.sum_gradients_all_reduce(
                "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)))
        self.device_grads = [list(zip(*dev_gv))[0] for dev_gv in self.device_grads_and_vars]

        assert(len(self.device_grads_and_vars) == num_devices)
        assert(len(self.device_grads_and_vars[0]) == 314)
        assert(len(self.device_grads_and_vars[0][0]) == 2)

        self.apply_op = tf.group(
            *[m.optimizer.apply_gradients(g) for g, m in zip(self.device_grads_and_vars, models)])
        self.sess.run(tf.global_variables_initializer())

    def feed_dict(self):
        result = {}
        for m in self.models:
            result.update(m.feed_dict())
        return result 

    def compute_apply(self, write_timeline):
       if write_timeline:
           run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
           run_metadata = tf.RunMetadata()
           self.sess.run(self.apply_op, feed_dict=self.feed_dict(),
               options=run_options,
               run_metadata=run_metadata)
           trace = timeline.Timeline(step_stats=run_metadata.step_stats)
           outf = "timeline-sgd.json"
           trace_file = open(outf, "w")
           print("wrote tf timeline to", os.path.abspath(outf))
           trace_file.write(trace.generate_chrome_trace_format())
       else:
          self.sess.run(self.apply_op, feed_dict=self.feed_dict())

    def compute_gradients(self):
        fetches = self.sess.run(
            self.device_grads,
            feed_dict=self.feed_dict())
        return fetches[0]

    def apply_gradients(self, avg_grads):
        result = {}
        for device_grads_and_vars in self.device_grads_and_vars:
            m = {device_grads_and_vars[j][0]: grad for j, grad in enumerate(avg_grads)}
            result.update(m)
        self.sess.run(self.apply_op, feed_dict=result)


def average_gradients(grads):
    out = []
    for grad_list in zip(*grads):
        out.append(np.mean(grad_list, axis=0))
    return out


def do_sgd_step(actors, local_only, write_timeline):
    if local_only:
        ray.get([a.compute_apply.remote(write_timeline) for a in actors])
    else:
        grads = ray.get([a.compute_gradients.remote() for a in actors])
        if len(actors) == 1:
            assert len(grads) == 1
            avg_grad = grads[0]
        else:
            # TODO(ekl) replace with allreduce
            avg_grad = average_gradients(grads)
        for a in actors:
            a.apply_gradients.remote(avg_grad)


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
parser.add_argument("--local-only", action="store_true",
    help="Whether to skip the object store for performance testing.")
parser.add_argument("--use-cpus", action="store_true",
    help="Whether to use CPU devices instead of GPU for debugging.")
parser.add_argument("--batch-size", type=int, default=64,
    help="ResNet101 batch size")


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    if args.timeline:
        assert args.local_only
    model = TFBenchModel
    if args.use_cpus:
        requests = {"num_cpus": args.devices_per_actor}
    else:
        requests = {"num_gpus": args.devices_per_actor}
    RemoteSGDWorker = ray.remote(**requests)(SGDWorker)
    actors = [
        RemoteSGDWorker.remote(
            i, model, args.batch_size, args.use_cpus and "xring" or "nccl",
            use_cpus=args.use_cpus, num_devices=args.devices_per_actor)
        for i in range(args.num_actors)]
    print("Test config: " + str(args))
    for i in range(10):
        start = time.time()
        print("Distributed sgd step", i)
        do_sgd_step(actors, args.local_only, args.timeline)
        print("Images per second", args.batch_size * args.num_actors * args.devices_per_actor / (time.time() - start))
