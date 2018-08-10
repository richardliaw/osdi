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
from allreduce import AllreduceRing as AllReduceActor
import os
import ray
import time
import pyarrow.plasma as plasma


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
parser.add_argument("--allreduce", action="store_true",
    help="Whether to use ring allreduce")
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


if __name__ == "__main__":
    import sys
    sys.exit(1)
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

    print("Test config: " + str(args))
    results = []

    if args.allreduce:
        print("Waiting for gradient configuration")
        shard_shapes = ray.get(actors[0].shard_shapes.remote())

        print("Making sure sgd actors start...")
        ray.get([a.shard_shapes.remote() for a in actors])

        print("Creating allreduce actors")
        allreduce_actors = create_allreduce_actors(actors, shard_shapes)
        assert len(allreduce_actors) == len(shard_shapes)

        print("Verify actor colocation")
        for a_list in allreduce_actors:
            assert len(a_list) == len(actors)
            for a, b in zip(a_list, actors):
                a_ip, b_ip = ray.get([a.ip.remote(), b.ip.remote()])
                assert a_ip == b_ip, (a_ip, b_ip)

        print("Warming up nodes")
        if args.warmup:
            ray.get([a.warmup.remote() for a in actors])

        step_fn = lambda: allreduce_sgd_step(actors, allreduce_actors, shard_shapes, args)

    elif args.ps:

        print("Waiting for gradient configuration")
        shard_shapes = ray.get(actors[0].shard_shapes.remote())

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
        step_fn = lambda: distributed_sgd_step(actors, ps_list, args)

    else:
        assert args.num_actors == 1
        step_fn = lambda: do_sgd_step(actors, args)

    for i in range(100):
        start = time.time()
        print("Sgd step", i)
        step_fn()
        ips = args.batch_size * args.num_actors * (args.override_devices or args.devices_per_actor) / (time.time() - start)
        print("Images per second", ips)
        if i > 3:
            results.append(ips)

    print("Mean, Median, Max IPS", np.mean(results[:10]), np.median(results[:10]), np.max(results[:10]))
    print("Mean, Median, Max IPS", np.mean(results[:15]), np.median(results[:15]), np.max(results[:15]))
    print("Mean, Median, Max IPS", np.mean(results), np.median(results), np.max(results))
