#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.nccl as nccl
from tensorflow.python.client import timeline
from tfbench import model_config, allreduce
from filelock import FileLock
from chrome_timeline import Timeline
import os
import ray
import time


def fetch(oids):
    for o in oids:
        plasma_id = ray.pyarrow.plasma.ObjectID(o)
        print("starting fetch")
        ray.worker.global_worker.plasma_client.fetch([plasma_id])
        print("finished fetch")


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


class MockDataset():
    name = "synthetic"


class TFBenchModel(object):
    def __init__(self, batch=64, use_cpus=False, device=""):
        image_shape = [batch, 224, 224, 3]
        labels_shape = [batch]
        self.device = device

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
                 use_xray=True,
                 plasma_op=False,
                 verbose=False):
        # TODO - just port VariableMgrLocalReplicated
        if use_xray:
            if num_devices == 4:
                gpu0 = FileLock("/tmp/gpu0")
                gpu1 = FileLock("/tmp/gpu1")
                try:
                    gpu0.acquire(timeout=0)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                except:
                    gpu1.acquire(timeout=0)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
                print("CUDA VISIBLES", os.environ["CUDA_VISIBLE_DEVICES"])
        self.i = i
        assert num_devices > 0
        tf_session_args = {
            "device_count": {"CPU": num_devices},
            "log_device_placement": False,
            "gpu_options": tf.GPUOptions(force_gpu_compatible=True),
            "inter_op_parallelism_threads": 128,
        }
        config_proto = tf.ConfigProto(**tf_session_args)
        self.sess = tf.Session(config=config_proto)
        models = []
        grad_ops = []
        self.iter = 0

        if use_cpus:
            device_tmpl = "/cpu:%d"
        else:
            device_tmpl = "/gpu:%d"
        for device_idx in range(num_devices):
            device = device_tmpl % device_idx
            with tf.device(device):
                with tf.variable_scope("device_%d" % device_idx):
                    print("DEVICE: ", device)
                    model = model_cls(batch=batch_size, use_cpus=use_cpus, device=device)
                    models += [model]
                    model.grads = [
                        t for t in model.optimizer.compute_gradients(model.loss)
                        if t[0] is not None]
                    grad_ops.append(model.grads)

        self.models = models
        if num_devices == 1:
           assert not max_bytes, "Not supported with 1 GPU"
           self.packed_grads_and_vars = grad_ops
        elif all_reduce_alg:
            if max_bytes:
                from tfbench import modified_allreduce
                self.packed_grads_and_vars, packing_vals = modified_allreduce.sum_gradients_all_reduce(
                    "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)),
                    agg_small_grads_max_bytes=max_bytes)
            else:
                self.packed_grads_and_vars = allreduce.sum_gradients_all_reduce(
                    "", grad_ops, 1, all_reduce_alg, 1, list(range(num_devices)))
        self.per_device_grads = [list(zip(*dev_gv))[0] for dev_gv in self.packed_grads_and_vars]
        assert(len(self.per_device_grads) == num_devices)
        self.num_grads = num_grads = len(self.packed_grads_and_vars[0])
        if max_bytes:
            assert(num_grads < 314)
            print("Packed grads => {} tensors".format(num_grads))
        else:
            assert(num_grads == 314)

        # Ops for reading grads with the right control deps
        nccl_noops = []
        for j in range(num_grads)[::-1]:
            with tf.control_dependencies(nccl_noops + [dev_grad[j] for dev_grad in self.per_device_grads]):
                nccl_noops = [tf.no_op()]

        # You must fetch this otherwise the NCCL allreduce will hang
        self.nccl_control_out = tf.group(*nccl_noops)

        round_robin_devices = False
        if args.plasma_op:
            memcpy_plasma_module = tf.load_op_library("/home/ubuntu/osdi2018/ops/memcpy_plasma_op.so")

            # For fetching grads -> plasma
            self.plasma_in_grads = []
            self.plasma_in_grads_oids = [
                tf.placeholder(shape=[], dtype=tf.string) for _ in range(num_grads)]
            ix = 0
            for j in range(num_grads):
                grad = self.per_device_grads[ix][j]
                if round_robin_devices:
                    ix += 1  # round robin assignment
                ix %= num_devices
                with tf.device(self.models[ix].device):
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
            packed_plasma_grads = []
            ix = 0
            for j in range(num_grads):
                with tf.device(self.plasma_in_grads[j].device):
                    with tf.control_dependencies([self.plasma_in_grads[j]]):
                        grad_ph = memcpy_plasma_module.plasma_to_tensor(
                            self.plasma_out_grads_oids[j],
                            plasma_store_socket_name=ray.worker.global_worker.plasma_client.store_socket_name,
                            plasma_manager_socket_name=ray.worker.global_worker.plasma_client.manager_socket_name)
                grad_ph = tf.reshape(grad_ph, self.packed_grads_and_vars[0][j][0].shape)
                print("Packed tensor", grad_ph)
                packed_plasma_grads.append(grad_ph)
            for i in range(num_devices):
                per_device = []
                for j, (g, v) in enumerate(self.packed_grads_and_vars[i]):
                    grad_ph = packed_plasma_grads[j]
                    per_device.append((grad_ph, v))
                unpacked_gv.append(per_device)

            if max_bytes:
                unpacked_gv = allreduce.unpack_small_tensors(unpacked_gv, packing_vals)

        elif max_bytes:
            unpacked_gv = allreduce.unpack_small_tensors(self.packed_grads_and_vars, packing_vals)
        else:
            unpacked_gv = self.packed_grads_and_vars

        # Same shape as packed_grads_and_vars
        assert len(unpacked_gv) == num_devices
        assert len(unpacked_gv[0]) == 314
        assert len(unpacked_gv[0][0]) == 2

        apply_ops = []
        to_apply = unpacked_gv[0]
        for ix, m in enumerate(models):
            apply_ops.append(m.optimizer.apply_gradients(
                [(g, v) for ((g, _), (_, v)) in zip(to_apply, unpacked_gv[ix])]))
        self.apply_op = tf.group(*apply_ops)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def compute_apply(self, args):
        run_timeline(
            self.sess, [self.apply_op, self.nccl_control_out],
            write_timeline=args.timeline, name="compute_apply")

    def compute_apply_split(self, args):
        grad = self.compute_gradients(args)
        self.apply_gradients(grad, args)

    def compute_gradients(self, args):
        start = time.time()
        fetches = self.sess.run(list(self.per_device_grads[0]) + [self.nccl_control_out])
        if args.verbose:
            print("compute grad interior time", time.time() - start)
        return fetches

    def apply_gradients(self, avg_grads, args):
        start = time.time()
        result = {
            g: avg_grads[i] for (i, g) in enumerate(self.per_device_grads[0])
        }
        self.sess.run(self.apply_op, feed_dict=result)
        if args.verbose:
            print("apply grad interior time", time.time() - start)

    def compute_apply_plasma(self, args):
        plasma_oids = [
            np.random.bytes(20) for _ in self.plasma_in_grads_oids]
        feed_dict = {
            ph: oid
            for (ph, oid) in zip(self.plasma_in_grads_oids, plasma_oids)
        }
        feed_dict.update({
            ph: oid
            for (ph, oid) in zip(self.plasma_out_grads_oids, plasma_oids)
        })
        run_timeline(
            self.sess, [self.plasma_in_grads, self.apply_op, self.nccl_control_out],
            feed_dict=feed_dict,
            write_timeline=args.timeline, name="compute_apply_plasma")

    def ps_compute_apply(self, out_grad_shard_oids, agg_grad_shard_oids):
        feed_dict = {
            ph: oid
            for (ph, oid) in zip(self.plasma_in_grads_oids, out_grad_shard_oids)
        }
        feed_dict.update({
            ph: oid
            for (ph, oid) in zip(self.plasma_out_grads_oids, agg_grad_shard_oids)
        })
        fetch(agg_grad_shard_oids)
        self.iter += 1
        run_timeline(
            self.sess, [self.plasma_in_grads, self.apply_op, self.nccl_control_out],
            feed_dict=feed_dict,
            write_timeline=args.timeline or self.iter == 2, name="ps_compute_apply")

    def compute_gradients_to_plasma_direct(self, args):
        plasma_in_grads_oids = [
            np.random.bytes(20) for _ in self.plasma_in_grads_oids]
        start = time.time()
        run_timeline(self.sess, self.plasma_in_grads + [self.nccl_control_out], feed_dict={
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
        run_timeline(
            self.sess, self.apply_op, feed_dict=feed,
            write_timeline=args.timeline, name="apply_plasma_direct")
        if args.verbose:
            print("apply grad plasma interior time", time.time() - start)

    def num_grad_shards(self):
        return self.num_grads

    def shard_shapes(self):
        main_gv = self.packed_grads_and_vars[0]
        return [g.shape for g, _ in main_gv]

    def ip(self):
        return ray.services.get_node_ip_address()


class ParameterServer(object):
    def __init__(self, num_workers, tid):
        self.num_sgd_workers = num_workers
        self.acc_counter = 0
        self.timeline = Timeline(tid)
        self.timeline.patch_ray()

    def set_tid(self, tid):
        self.timeline.tid = tid

    def get_time(self):
        return time.time() + self.timeline.offset

    def set_time(self, ref_time):
        self.timeline.offset = ref_time - time.time()

    def initialize(self, shard_shape):
        self.accumulated = np.zeros(shard_shape, dtype=np.float32)

    def warmup(self):
        warmup()

    def mark(self):
        self.timeline.event("mark")

    def prefetch(self, oids):
        self.timeline.reset()
        self.timeline.start("prefetch")
        fetch(oids)
        self.timeline.end("prefetch")

    def add_spinwait(self, grad_shard_ids):
        self.timeline.start("add_spinwait")
        plasma_ids = [ray.pyarrow.plasma.ObjectID(x) for x in grad_shard_ids]
        while plasma_ids:
            for p in plasma_ids:
                if ray.worker.global_worker.plasma_client.contains(p):
                    self.timeline.start("get_buffers")
                    [raw_grads] = ray.worker.global_worker.plasma_client.get_buffers([p])
                    grads = np.frombuffer(raw_grads, dtype=np.float32)
                    self.accumulated += grads
                    self.acc_counter += 1
                    self.timeline.end("get_buffers")
                    plasma_ids.remove(p)
                    break
        self.timeline.end("add_spinwait")

    def add(self, grad_shard_id):
        self.timeline.start("add")
        # self.timeline.start("add_wait")
        # ray.wait([ray.local_scheduler.ObjectID(grad_shard_id)])
        # self.timeline.end("add_wait")
        self.timeline.start("get_buffers")
        oid = ray.pyarrow.plasma.ObjectID(grad_shard_id)
        [raw_grads] = ray.worker.global_worker.plasma_client.get_buffers([oid])
        grads = np.frombuffer(raw_grads, dtype=np.float32)
        self.timeline.end("get_buffers")
        self.accumulated += grads
        self.acc_counter += 1
        self.timeline.end("add")

    def get(self, object_id):
        self.timeline.start("get")
        client = ray.worker.global_worker.plasma_client
        assert self.acc_counter == self.num_sgd_workers, self.acc_counter
        oid = ray.pyarrow.plasma.ObjectID(object_id)
        buff = client.create(
            oid, self.accumulated.nbytes)
        wrapper = np.frombuffer(buff, dtype=np.float32)
        np.copyto(wrapper, self.accumulated)
        client.seal(oid)
        self.accumulated = np.zeros_like(self.accumulated)
        self.acc_counter = 0
        self.timeline.end("get")

    def wait_for_grads(self, grad_shard_ids):
        plasma_ids = [(i, ray.pyarrow.plasma.ObjectID(x)) for (i, x) in enumerate(grad_shard_ids)]
        start = time.time()
        while plasma_ids and time.time() - start < 5:
            for (i, p) in plasma_ids:
                if ray.worker.global_worker.plasma_client.contains(p):
                    self.timeline.event("grad_{}_arrived".format(i))
                    plasma_ids.remove((i, p))
                    break

    def get_timeline(self):
        return self.timeline

    def ip(self):
        return ray.services.get_node_ip_address()

    def pin(self, cpu_id):
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([cpu_id])
            print("Setting CPU Affinity to: ", cpu_id)
        except Exception as e:
            print(e)


def average_gradients(grads):
    out = []
    for grad_list in zip(*grads):
        out.append(np.mean(grad_list, axis=0))
    return out


def do_sgd_step(actors, args):
    if args.local_only:
        if args.plasma_op:
            ray.get([a.compute_apply_plasma.remote(args) for a in actors])
        elif args.split:
            ray.get([a.compute_apply_split.remote(args) for a in actors])
        else:
            ray.get([a.compute_apply.remote(args) for a in actors])
    else:
        assert not args.split
        start = time.time()
        if args.plasma_op:
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
        if args.plasma_op:
            ray.get([a.apply_gradients_from_plasma_direct.remote(avg_grad, args) for a in actors])
        else:
            ray.get([a.apply_gradients.remote(avg_grad, args) for a in actors])
        if args.verbose:
            print("apply all grads time", time.time() - start)


def distributed_sgd_step(actors, ps_list, args):
    # Preallocate object ids that actors will write gradient shards to
    grad_shard_oids_list = [
        [np.random.bytes(20) for _ in ps_list]
        for _ in actors
    ]
    print("generated grad oids")

    # Preallocate object ids that param servers will write new weights to
    accum_shard_ids = [np.random.bytes(20) for _ in ps_list]
    print("generated accum oids")

    # Kick off the fused compute grad / update weights tf run for each actor
    for actor, grad_shard_oids in zip(actors, grad_shard_oids_list):
        actor.ps_compute_apply.remote(grad_shard_oids, accum_shard_ids)
    print("Launched all ps_compute_applys on all actors")

    # Issue prefetch ops
    for j, (ps, weight_shard_oid) in list(enumerate(zip(ps_list, accum_shard_ids)))[::-1]:
        to_fetch = []
        for grad_shard_oids in grad_shard_oids_list:
            to_fetch.append(grad_shard_oids[j])
        random.shuffle(to_fetch)
        ps.prefetch.remote(to_fetch)
    print("Launched all prefetch ops")

    # Aggregate the gradients produced by the actors. These operations
    # run concurrently with the actor methods above.
    ps_gets = []
    for j, (ps, weight_shard_oid) in list(
            enumerate(zip(ps_list, accum_shard_ids)))[::-1]:
        if args.ps_spinwait:
            ps.add_spinwait.remote([gs[j] for gs in grad_shard_oids_list])
        else:
            for grad_shard_oids in grad_shard_oids_list:
                ps.add.remote(grad_shard_oids[j])
        ps_gets.append(ps.get.remote(weight_shard_oid))
    print("Launched all aggregate ops")

    if args.debug_ps:
        for ps in ps_list:
            ps.wait_for_grads.remote(accum_shard_ids)
        print("Launched debug ops")

    if args.verbose:
        timelines = [ps.get_timeline.remote() for ps in ps_list]
        print("launched timeline gets")
        timelines = ray.get(timelines)
        t0 = timelines[0]
        for t in timelines[1:]:
            t0.merge(t)
        t0.chrome_trace_format("ps_timeline.json")
    else:
        # Wait for at least the ps gets to finish
        ray.get(ps_gets)


import argparse

parser = argparse.ArgumentParser()

# Scaling
parser.add_argument("--devices-per-actor", type=int, default=1,
    help="Number of GPU/CPU towers to use per actor")
parser.add_argument("--override-devices", type=int, default=0,
    help="Number of GPU/CPU towers to use per actor for real")
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
parser.add_argument("--ps", action="store_true",
    help="Whether to use param server")
parser.add_argument("--split", action="store_true",
    help="Whether to split compute and apply in local only mode.")
parser.add_argument("--plasma-op", action="store_true",
    help="Whether to use the plasma TF op.")
parser.add_argument("--ps-spinwait", action="store_true",
    help="Whether to spin wait for plasma to download objects")
parser.add_argument("--cluster", action="store_true",
    help="Whether to use a Ray cluster")
parser.add_argument("--roundrobin_ps", action="store_true",
    help="Whether to round robin place PS shards. Requires cluster to be true"
         "and each node to only hae one actor")
parser.add_argument("--spread_ps", action="store_true",
    help="Whether to force PS to be allocated on nodes other than SGD workers")
parser.add_argument("--debug_ps", action="store_true",
    help="Whether to add debug markers for timeline")
parser.add_argument("--use-cpus", action="store_true",
    help="Whether to use CPU devices instead of GPU for debugging.")
parser.add_argument("--set-visible-devs", action="store_false",
    help="Whether to set visible devices. Defaults to True; needed for x-ray.")
parser.add_argument("--max-bytes", type=int, default=0,
    help="Max byte tensor to pack")
parser.add_argument("--batch-size", type=int, default=64,
    help="ResNet101 batch size")
parser.add_argument("--allreduce-spec", type=str, default="simple",
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


def roundrobin_ps(ps_cls, sgd_workers, shard_shapes, spread_ps):
    worker_ips = ray.get([w.ip.remote() for w in sgd_workers])
    num_ips = len(set(worker_ips))
    num_workers = len(sgd_workers)
    min_placed = np.ceil(len(shard_shapes) / num_ips)
    from collections import Counter, defaultdict
    tid_counter = [0]

    def create_ps():
        tid_counter[0] += 1
        time.sleep(1)  # needed because resource tracking is faulty
        return RemotePS.remote(num_workers, tid_counter[0])

    ip_mapping = defaultdict(list)

    while (any(len(v) < min_placed for v in ip_mapping.values())
              or (len(ip_mapping) < num_ips)):
        print("generating new ps, ip map so far", ip_mapping)
        new_ps = create_ps()
        ps_ip = ray.get(new_ps.ip.remote())
        if spread_ps and ps_ip in worker_ips:
            print("ignoring ps that is on same node as worker")
        elif not spread_ps and ps_ip not in worker_ips:
            print("ignoring ps that NOT on same node as some worker")
        else:
            ip_mapping[ps_ip] += [new_ps]

    final_list = []
    candidates = list(ip_mapping.values())
    for i, s in enumerate(shard_shapes):
        ps = candidates[i % num_ips][i // num_ips]
        final_list += [ps]
        ps.initialize.remote(s)

    for ps in sum(candidates, []):
        if ps not in final_list:
            ps.__ray_terminate__.remote(ps._ray_actor_id.id())
            print("removing a ps...")
        else:
            print("saving ps...")

    print("Final PS balance: ", Counter(ray.get([ps.ip.remote() for ps in final_list])))
    for i, ps in enumerate(final_list):
        ps.set_tid.remote(i)
    return final_list


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cluster:
        redis_address = "localhost:6379"
    else:
        redis_address = None
    if args.hugepages:
        ray.init(huge_pages=True, plasma_directory="/mnt/hugepages/", redis_address=redis_address)
    else:
        ray.init(redirect_output=False, redis_address=redis_address, use_raylet=True)
    if args.warmup and not args.ps:
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
        spec = args.allreduce_spec
    actors = []
    for i in range(args.num_actors):
        actors += [RemoteSGDWorker.remote(
            i, model, args.batch_size, spec,
            use_cpus=args.use_cpus, num_devices=args.override_devices or args.devices_per_actor,
            max_bytes=args.max_bytes, plasma_op=args.plasma_op,
            verbose=args.verbose)]
        print("Creating an actor")
        time.sleep(1)

    print("Test config: " + str(args))
    if args.ps:
        print("Waiting for gradient configuration")
        shard_shapes = ray.get(actors[0].shard_shapes.remote())
        results = []
        print("making sure actors start...")
        ray.get([a.shard_shapes.remote() for a in actors])
        print("all actors started")
        RemotePS = ray.remote(ParameterServer)
        if args.roundrobin_ps:
            print("## !! Round Robin Assumes Each Node only has 1 SGDWorker !!")
            assert args.cluster
            assert len(actors) > 1, "Need more than 1 node for round robin!"
            ps_list = roundrobin_ps(RemotePS, actors, shard_shapes, args.spread_ps)
        else:
            ps_list = [RemotePS.remote(len(actors), i) 
                       for i, s in enumerate(shard_shapes)]
            [ps.initialize.remote(s) for ps, s in zip(ps_list, shard_shapes)]
        print("All PS started")
        for _ in range(10):
            [a.set_time.remote(time.time()) for a in ps_list]
            times = ray.get([a.get_time.remote() for a in ps_list])
        print("Clock skew ms: " + str((max(times) - min(times)) * 1000))
        if args.warmup:
            ray.get([ps.warmup.remote() for ps in ps_list])
        print("All PS warmed")
        for i in range(20):
            start = time.time()
            print("PS sgd step", i)
            distributed_sgd_step(actors, ps_list, args)
            ips = args.batch_size * args.num_actors * (args.override_devices or args.devices_per_actor) / (time.time() - start)
            print("Images per second", ips)
            if i > 3:
                results.append(ips)
        print("Mean, Median, Max IPS", np.mean(results), np.median(results), np.max(results))
    else:
        assert args.num_actors == 1
        for i in range(10):
            start = time.time()
            print("Local sgd step", i)
            do_sgd_step(actors, args)
            print("Images per second", args.batch_size * args.num_actors * (args.override_devices or args.devices_per_actor) / (time.time() - start))
