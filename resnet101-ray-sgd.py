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


def run_timeline(sess, ops, feed_dict={}, write_timeline=False, name=""):
    if write_timeline:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fetches = sess.run(
            ops, options=run_options, run_metadata=run_metadata,
            feed_dict=feed_dict)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        outf = "timeline-{}.json".format(name)
        trace_file = open(outf, "w")
        print("wrote tf timeline to", os.path.abspath(outf))
        trace_file.write(trace.generate_chrome_trace_format())
    else:
        fetches = sess.run(ops, feed_dict=feed_dict)
    return fetches


class MockDataset():
    name = "synthetic"


class TFBenchModel(object):
    def __init__(self, batch=64, use_cpus=False):
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

        self.model = model_config.get_model_config("vgg11", MockDataset())
        logits, aux = self.model.build_network(self.inputs, data_format=use_cpus and "NHWC" or "NCHW")
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
                 plasma_op=False,
                 verbose=False):
        # TODO - just port VariableMgrLocalReplicated
        self.i = i
        assert num_devices > 0
        tf_session_args = {
            "device_count": {"CPU": num_devices},
            "log_device_placement": False,
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
                    model = model_cls(batch=batch_size, use_cpus=use_cpus)
                    models += [model]
                    model.grads = [
                        t for t in model.optimizer.compute_gradients(model.loss)
                        if t[0] is not None]
                    grad_ops.append(model.grads)

        self.models = models
        if num_devices == 1:
           assert not max_bytes, "Not supported with 1 GPU"
           self.per_device_grads_and_vars = grad_ops
        elif all_reduce_alg:
            if max_bytes:
                import ipdb; ipdb.set_trace()
                from tfbench import modified_allreduce
                self.per_device_grads_and_vars, packing_vals = modified_allreduce.sum_gradients_all_reduce(
                    "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)),
                    agg_small_grads_max_bytes=max_bytes,
                    agg_small_grads_max_group=9999)
            else:
                self.per_device_grads_and_vars = allreduce.sum_gradients_all_reduce(
                    "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)))
        self.per_device_grads = [list(zip(*dev_gv))[0] for dev_gv in self.per_device_grads_and_vars]
        assert(len(self.per_device_grads) == num_devices)
        num_grads = len(self.per_device_grads_and_vars[0])
        if max_bytes:
            assert(num_grads < 314)
            print("Packed grads => {} tensors".format(num_grads))
        else:
            assert(num_grads == 314)

        if plasma_op:
            memcpy_plasma_module = tf.load_op_library("ops/memcpy_plasma_op.so")

            # For fetching grads -> plasma
            self.plasma_in_grads = []
            self.plasma_in_grads_oids = [
                tf.placeholder(shape=[], dtype=tf.string) for _ in range(num_grads)]
            for j, grad in enumerate(self.per_device_grads[0]):  # from 0th device
                with tf.control_dependencies([dev_grad[j] for dev_grad in self.per_device_grads]):
                    plasma_grad = memcpy_plasma_module.tensor_to_plasma(
                        [grad],
                        self.plasma_in_grads_oids[j],
                        plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                        plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name)
                    self.plasma_in_grads.append(plasma_grad)

            # For applying grads <- plasma
            unpacked_gv = []
            self.plasma_out_grads_oids = [
                tf.placeholder(shape=[], dtype=tf.string) for _ in range(num_grads)]
            for i in range(num_devices):
                per_device = []
                for j, (g, v) in enumerate(self.per_device_grads_and_vars[i]):
                    grad_ph = memcpy_plasma_module.plasma_to_tensor(
                        self.plasma_out_grads_oids[j],
                        plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                        plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name)
                    grad_ph = tf.reshape(grad_ph, g.shape)
                    print("Packed tensor", grad_ph)
                    per_device.append((grad_ph, v))
                unpacked_gv.append(per_device)

            if max_bytes:
                unpacked_gv = allreduce.unpack_small_tensors(unpacked_gv, packing_vals)

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

    def compute_apply(self, args):
        run_timeline(self.sess, self.apply_op, write_timeline=args.timeline, name="compute_apply")

    def compute_apply_split(self, args):
        grad = self.compute_gradients(args)
        self.apply_gradients(grad, args)

    def compute_gradients(self, args):
        start = time.time()
        fetches = self.sess.run(self.per_device_grads)
        if args.verbose:
            print("compute grad interior time", time.time() - start)
        return fetches[0]

    def apply_gradients(self, avg_grads, args):
        start = time.time()
        result = {}
        for per_device_grads_and_vars in self.per_device_grads_and_vars:
            m = {per_device_grads_and_vars[j][0]: grad for j, grad in enumerate(avg_grads)}
            result.update(m)
        self.sess.run(self.apply_op, feed_dict=result)
        if args.verbose:
            print("apply grad interior time", time.time() - start)

    def compute_gradients_to_plasma_direct(self, args):
        plasma_in_grads_oids = [
            np.random.bytes(20) for _ in self.plasma_in_grads_oids]
        start = time.time()
        fetches = run_timeline(self.sess, self.plasma_in_grads, feed_dict={
            ph: oid for (ph, oid) in zip(self.plasma_in_grads_oids, plasma_in_grads_oids)
        }, write_timeline=args.timeline, name="grads_plasma_direct")
        if args.verbose:
            print("compute grad plasma interior time", time.time() - start)
        return plasma_in_grads_oids

    def apply_gradients_from_plasma_direct(self, avg_grads_oids, args):
        start = time.time()
        feed = {
            ph: oid for (ph, oid) in zip(self.plasma_out_grads_oids, avg_grads_oids)
        }
        self.sess.run(self.apply_op, feed_dict=feed)
        if args.verbose:
            print("apply grad plasma interior time", time.time() - start)


def average_gradients(grads):
    out = []
    for grad_list in zip(*grads):
        out.append(np.mean(grad_list, axis=0))
    return out


def do_sgd_step(actors, args):
    if args.local_only:
        if args.split:
            ray.get([a.compute_apply_split.remote(args) for a in actors])
        else:
            ray.get([a.compute_apply.remote(args) for a in actors])
    else:
        assert not args.split
        start = time.time()
        if plasma_op:
            grads = ray.get([a.compute_gradients_to_plasma_direct.remote(args) for a in actors])
        else:
            grads = ray.get([a.compute_gradients.remote(args) for a in actors])
        if args.verbose:
            print("compute all grads time", time.time() - start)
        start = time.time()
        if len(actors) == 1:
            assert len(grads) == 1
            avg_grad = grads[0]
        else:
            # TODO(ekl) replace with allreduce
            avg_grad = average_gradients(grads)
        if args.verbose:
            print("distributed allreduce time", time.time() - start)
        start = time.time()
        if plasma_op:
            print("TODO apply grads crashes with plasma op, skipping for now")
#            ray.get([a.apply_gradients_from_plasma_direct.remote(avg_grad, args) for a in actors])
        else:
            ray.get([a.apply_gradients.remote(avg_grad, args) for a in actors])
        if args.verbose:
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
parser.add_argument("--split", action="store_true",
    help="Whether to split compute and apply in local only mode.")
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
            max_bytes=args.max_bytes, plasma_op=args.plasma_op,
            verbose=args.verbose)
        for i in range(args.num_actors)]
    print("Test config: " + str(args))
    for i in range(10):
        start = time.time()
        print("Distributed sgd step", i)
        do_sgd_step(actors, args)
        print("Images per second", args.batch_size * args.num_actors * args.devices_per_actor / (time.time() - start))
