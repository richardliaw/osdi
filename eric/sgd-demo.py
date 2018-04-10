from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tfbench import model_config
import ray



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
        self.model = model_config.get_model_config("resnet50", MockDataset())
        logits, aux = self.model.build_network(self.inputs, data_format="NHWC")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='xentropy-loss')

    def feed_dict(self):
        return {
            self.inputs: np.zeros(self.inputs.shape.as_list()),
            self.labels: np.zeros(self.labels.shape.as_list())
        }


@ray.remote
class SGDWorker(object):
    def __init__(self, i, model_cls):
        self.i = i
        self.sess = tf.Session()
        self.model = model_cls()
        self.optimizer = tf.train.AdamOptimizer()
        self.grad_op = self.optimizer.compute_gradients(self.model.loss)
        self.apply_op = self.optimizer.apply_gradients(self.grad_op)
        self.sess.run(tf.global_variables_initializer())

    def compute_gradients(self):
        l, g = self.sess.run(
            [self.model.loss, [g[0] for g in self.grad_op]],
            feed_dict=self.model.feed_dict())
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
def do_sgd_step(actors):
    grads = ray.get([a.compute_gradients.remote() for a in actors])
    print("Avg loss", np.mean([l for (l, g) in grads]))
    avg_grad = average_gradients([g for (l, g) in grads])
    for a in actors:
        a.apply_gradients.remote(avg_grad)


if __name__ == "__main__":
    ray.init()
    # from resnet_demo import RayModel

    model = TFBenchModel
    actors = [SGDWorker.remote(i, model) for i in range(2)]
    for i in range(100):
        print("Distributed sgd step", i)
        do_sgd_step(actors)
