from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tfbench import model_config, allreduce
import ray
import time


class SimpleModel(object):
    def __init__(self):
        self.inputs = tf.placeholder(shape=[None, 4], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        prediction = slim.fully_connected(
            self.inputs, 1, scope="layer1")
        self.loss = tf.reduce_mean(
            tf.squared_difference(prediction, self.labels))

    def feed_dict(self):
        return {
            self.inputs: np.ones((32, 4)),
            self.labels: np.zeros((32, 2)),
        }

class MockDataset():
    name = "synthetic"

class TFBenchModel(object):
    def __init__(self, batch=64):
        ## this is currently NHWC
        self.inputs = tf.placeholder(shape=[batch, 224, 224, 3, ], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[batch], dtype=tf.int32)
        self.model = model_config.get_model_config("resnet101", MockDataset())
        logits, aux = self.model.build_network(self.inputs, data_format="NHWC")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='xentropy-loss')
        self.optimizer = tf.train.AdamOptimizer()

    def feed_dict(self):
        return {
            self.inputs: np.zeros(self.inputs.shape.as_list()),
            self.labels: np.zeros(self.labels.shape.as_list())
        }
# import ipdb
# from tensorflow.python.client import timeline
# def debug():
#     return ipdb.set_trace(context=5)



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
           self.tower_gradvars = grad_ops
        elif all_reduce_alg:
           self.tower_gradvars = allreduce.sum_gradients_all_reduce(
                "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)))

        # for reading out to object store
        # self.avg_grad = self.individual_grads[0]
        # assert(len(self.avg_grad) == 314)
        # assert(len(self.avg_grad[0]) == 2)
        assert(len(self.tower_gradvars) == num_devices)
        assert(len(self.tower_gradvars[0]) == 314)
        assert(len(self.tower_gradvars[0][0]) == 2)
        self.tower_grads = [zip(*tower_gv)[0] for tower_gv in self.tower_gradvars]

        self.apply_op = tf.group(
            *[m.optimizer.apply_gradients(gv) for gv, m in zip(self.tower_gradvars, models)])
        self.sess.run(tf.global_variables_initializer())

    def feed_dict(self):
        result = {}
        for m in self.models:
            result.update(m.feed_dict())
        return result 

    def compute_apply(self):
       # import os
       # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
       # run_metadata = tf.RunMetadata()
       self.sess.run(self.apply_op, feed_dict=self.feed_dict(),)
       #     options=run_options,
       #     run_metadata=run_metadata)
       # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
       # trace_file = open("/tmp/timeline-load.json", "w")
       # trace_file.write(trace.generate_chrome_trace_format())

    def compute_gradients(self):
        """avg"""
        fetches = self.sess.run(
            self.tower_grads,
            feed_dict=self.feed_dict())
        # assert all fetches should be same
        tower_avg = fetches[0]  
        return tower_avg

    def apply_gradients(self, avg_grads):
        result = {}
        for device_grads in self.tower_grads:
            m = {device_grads[j]: grad for j, grad in enumerate(avg_grads)}
            result.update(m)
        self.sess.run(self.apply_op, feed_dict=result)


def average_gradients(grads):
    out = []
    for grad_list in zip(*grads):
        out.append(np.mean(grad_list, axis=0))
    return out


# TODO(ekl) replace with allreduce
def do_sgd_step(actors, skip_object_store):
    if skip_object_store:
        ray.get([a.compute_apply.remote() for a in actors])
    else:
        grads = ray.get([a.compute_gradients.remote() for a in actors])
        if len(actors) == 1:
            assert len(grads) == 1
            avg_grad = grads[0]
        else:
            avg_grad = average_gradients(grads)
        for a in actors:
            a.apply_gradients.remote(avg_grad)


import argparse

parser = argparse.ArgumentParser()

# Scaling
parser.add_argument("--devices-per-actor", type=int, default=2,
    help="Number of GPU/CPU towers to use per actor")
parser.add_argument("--num-actors", type=int, default=1,
    help="Number of actors to use for distributed sgd")

# Debug
parser.add_argument("--skip-plasma", action="store_true",
    help="Whether to skip the object store for performance testing.")
parser.add_argument("--use-cpus", action="store_true",
    help="Whether to use CPU devices instead of GPU for debugging.")
parser.add_argument("--batch-size", type=int, default=64,
    help="ResNet101 batch size")


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
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
        do_sgd_step(actors, args.skip_plasma)
        print("Images per second", args.batch_size * args.num_actors * args.devices_per_actor / (time.time() - start))
