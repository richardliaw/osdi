#!/usr/bin/env python

from resnet101_ray_sgd import ParameterServer
import random
import numpy as np
import time

import ray


class Shipper(object):
    def __init__(self):
        self.accumulated = np.random.rand(2500000).astype(np.float32)

    def seal(self, list_of_oids):
        print("starting seal")
        for object_id in list_of_oids:
            client = ray.worker.global_worker.plasma_client
            oid = ray.pyarrow.plasma.ObjectID(object_id)
            buff = client.create(oid, self.accumulated.nbytes)
            wrapper = np.frombuffer(buff, dtype=np.float32)
            np.copyto(wrapper, self.accumulated)
            client.seal(oid)
            print("sealed")
        return True


def distributed_sgd_step(actors, ps_list, args):
    # Preallocate object ids that actors will write gradient shards to
    grad_shard_oids_list = [[np.random.bytes(20) for _ in ps_list]
                            for _ in actors]
    print("generated grad oids")
    # Preallocate object ids that param servers will write new weights to
    accum_shard_ids = [np.random.bytes(20) for _ in ps_list]
    print("generated accum oids")

    seals = []
    print("starting seal")
    for a, grad_shard_list in zip(actors, grad_shard_oids_list):
        seals += [a.seal.remote(grad_shard_list)]

    ray.get(seals)
    print("Done sealing")

    print("Issue prefetch ops")
    for j, (ps, weight_shard_oid) in list(
            enumerate(zip(ps_list, accum_shard_ids)))[::-1]:
        to_fetch = []
        for grad_shard_oids in grad_shard_oids_list:
            to_fetch.append(grad_shard_oids[j])
        random.shuffle(to_fetch)
        ps.prefetch.remote(to_fetch)
    print("Launched all prefetch ops")

    # Aggregate the gradients produced by the actors. These operations
    # run concurrently with the actor methods above.
    for j, (ps, weight_shard_oid) in list(
            enumerate(zip(ps_list, accum_shard_ids)))[::-1]:
        for grad_shard_oids in grad_shard_oids_list:
            ps.add.remote(grad_shard_oids[j])
        ps.get.remote(weight_shard_oid)
    print("Launched all aggregate ops")

    timelines = [ps.get_timeline.remote() for ps in ps_list]
    print("launched timeline gets")

    if args.verbose:
        timelines = ray.get(timelines)
        t0 = timelines[0]
        for t in timelines[1:]:
            t0.merge(t)
        t0.chrome_trace_format("psbench_timeline.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Scaling
    parser.add_argument(
        "--devices-per-actor",
        type=int,
        default=1,
        help="Number of GPU/CPU towers to use per actor")
    parser.add_argument(
        "--num-actors",
        type=int,
        default=1,
        help="Number of actors to use for distributed sgd")

    # Debug
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out timing debug messages")
    args = parser.parse_args()
    if True or args.cluster:
        redis_address = "localhost:6379"
    else:
        redis_address = None
    ray.init(
        redirect_output=True, redis_address=redis_address, use_raylet=True)
    #if args.warmup:
    #    warmup()
    # model = TFBenchModel
    requests = {"num_gpus": args.devices_per_actor}
    RemoteShippers = ray.remote(**requests)(Shipper)
    RemotePS = ray.remote(ParameterServer)
    actors = []
    for i in range(args.num_actors):
        actors += [RemoteShippers.remote()]
        time.sleep(1)
    ps_list = [RemotePS.remote(2500000, args.num_actors, i) for i in range(15)]
    from collections import Counter
    print(Counter(ray.get([ps.ip.remote() for ps in ps_list])))
    for i in range(10):
        start = time.time()
        distributed_sgd_step(actors, ps_list, args)
        print("Took %f sec" % (time.time() - start))

