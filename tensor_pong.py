#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from chrome_timeline import Timeline
import os
import ray
import time


def fetch(oids):
    for o in oids:
        plasma_id = ray.pyarrow.plasma.ObjectID(o)
        ray.worker.global_worker.plasma_client.fetch([plasma_id])


def run_timeline(sess, ops, feed_dict={}, write_timeline=False, name=""):
    if write_timeline:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fetches = sess.run(
            ops, options=run_options, run_metadata=run_metadata,
            feed_dict=feed_dict)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        outf = "timeline-{}-{}.json".format(name, os.getpid())
        trace_file = open(outf, "w")
        print("wrote tf timeline to", os.path.abspath(outf))
        trace_file.write(trace.generate_chrome_trace_format())
    else:
        fetches = sess.run(ops, feed_dict=feed_dict)
    return fetches


@ray.remote(num_gpus=8)
class Actor(object):
    def __init__(self):
        self.sess = tf.Session()
        memcpy_plasma_module = tf.load_op_library("/home/ubuntu/osdi2018/ops/memcpy_plasma_op.so")

        self.plasma_oids = [
            tf.placeholder(shape=[], dtype=tf.string) for _ in range(100)]

        sends = []
        payload = tf.constant(np.random.bytes(1024 * 1024 * 10))
        for oid in self.plasma_oids:
            sends.append(
                memcpy_plasma_module.tensor_to_plasma(
                    [payload],
                    oid,
                    plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                    plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name))
        self.send = tf.group(*sends)

        recvs = []
        for oid in self.plasma_oids:
            recvs.append(
                memcpy_plasma_module.plasma_to_tensor(
                    oid,
                    plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                    plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name))
        self.recv = tf.group(*recvs)

    def sendall(self, oids):
        run_timeline(
            self.sess,
            self.send.op,
            feed_dict={ph: oid for (ph, oid) in zip(self.plasma_oids, oids)},
            write_timeline=True,
            name="sendall")

    def recvall(self, oids):
        run_timeline(
            self.sess,
            self.recv.op,
            feed_dict={ph: oid for (ph, oid) in zip(self.plasma_oids, oids)},
            write_timeline=True,
            name="recvall")


if __name__ == "__main__":
    a1 = Actor.remote()
    a2 = Actor.remote()

    oids = [np.random.bytes(20) for _ in range(100)]
    a1.sendall.remote(oids)
    ray.get(a2.recvall.remote(oids))
