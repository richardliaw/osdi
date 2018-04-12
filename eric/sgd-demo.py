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
    def __init__(self, i, model_cls, all_reduce_alg=None, num_gpus=1):
        # TODO - just port VariableMgrLocalReplicated
        self.i = i
        assert num_gpus > 0
        self.sess = tf.Session()
        models = []
        grad_ops = []

        for device_idx in range(num_gpus):
            with tf.device("/gpu:%d" % device_idx):
                with tf.variable_scope("device_%d" % device_idx):
                    model = model_cls()
                    models += [model]
                    model.grads = [t for t in model.optimizer.compute_gradients(model.loss) if t[0] is not None]
                    grad_ops.append(model.grads)

        self.models = models
        if num_gpus == 1:
           grad_ops = grad_ops[0] 
        elif all_reduce_alg:
            grad_ops = allreduce.sum_gradients_all_reduce(
                "", grad_ops, 1, all_reduce_alg, 1, list(range(num_gpus)))

        self.grad_op = grad_ops
        self.apply_op = tf.group(
            *[m.optimizer.apply_gradients(m.grads) for m in models])
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
           options=run_options,
           run_metadata=run_metadata)
       # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
       # trace_file = open("/tmp/timeline-load.json", "w")
       # trace_file.write(trace.generate_chrome_trace_format())

    def compute_gradients(self):
        l, g = self.sess.run(
            [self.models[0].loss, [g[0] for g in self.grad_op]],
            feed_dict=self.feed_dict())
        return l, g

    def apply_gradients(self, grads):
        feed_dict = {
            self.grad_op[i][0]: grads[i] for i in range(len(grads))
        }
        self.sess.run(self.apply_op, feed_dict=feed_dict)


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
        print("Avg loss", np.mean([l for (l, g) in grads]))
        if len(actors) == 1:
            avg_grad = grads[0][1]
        else:
            avg_grad = average_gradients([g for (l, g) in grads])
        for a in actors:
            a.apply_gradients.remote(avg_grad)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip_object_store", action="store_true")
parser.add_argument("--gpus_per_actor", type=int, default=1)
parser.add_argument("--num_actors", type=int, default=1)


if __name__ == "__main__":
    ray.init()
    model = TFBenchModel
    RemoteSGDWorker = ray.remote(num_gpus=args.gpus_per_actor)(SGDWorker)
    actors = [
        RemoteSGDWorker.remote(i, model, 'nccl', num_gpus=gpus) for i in range(args.num_actors)]
    for i in range(10):
        start = time.time()
        print("Distributed sgd step", i)
        do_sgd_step(actors, args.skip_object_store)
        print("Images per second", 64 * n * gpus / (time.time() - start))
