from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
from chrome_timeline import Timeline

import numpy as np
try:
    import torch
    TORCH_ENABLED = True
except:
    print("***unable to import torch***")
    TORCH_ENABLED = False
import ray

redis_address = os.environ.get('RAYREDIS', None)

RAYLET_LOCAL_MULTINODE=False


class Batch(object):
    """
    Simple object for creating an object that can
    generate batches of sequential integers.
    """
    def __init__(self, total_size, batch_size):
        """
        :param total_size: Total number of items to split into batches.
        :param batch_size: Size of each batch.
        """
        self.total_size = total_size
        self.batch_size = batch_size
        self.batches = self.get_batches(total_size, batch_size)
        self.num_batches = len(self.batches)

    def get_batches(self, total_size, batch_size):
        """
        :param total_size: Total number of items to split into batches.
        :param batch_size: Size of each batch.
        :return: A list of 2-tuples.
                 Each 2-tuple is a segment of indices corresponding to items of size batch_size.
                 The size of the list is total_size / batch_size.
        """
        if total_size < batch_size:
            return [[0, total_size]]
        batches = list(range(0, total_size, batch_size))
        num_batches = int(total_size / batch_size)
        batches = [batches[i:i + 2] for i in range(0, num_batches, 1)]
        if len(batches[-1]) == 1:
            batches[-1].append(total_size)
        if batches[-1][1] != total_size:
            batches.append([batches[-1][1], total_size])
        return batches


class WeightPartition(object):

    def __init__(self, worker_index, num_workers, buffer_size=None, buffer=None, dtype=np.float32):
        self.worker_index = worker_index
        self.num_workers = num_workers
        if buffer is not None:
            if buffer_size is None:
                buffer_size = len(buffer)
            else:
                assert buffer_size == len(buffer)
            self.buffer = buffer
            self.buffer_size = buffer_size
        elif buffer_size is not None:
            assert buffer is None
            self.buffer_size = buffer_size
            self.buffer = np.zeros(buffer_size, dtype=dtype)
        else:
            raise Exception("specify buffer or buffer_size.")
        self.batch_size = int(np.ceil(self.buffer_size/self.num_workers))
        self.batch_intervals = Batch(total_size=buffer_size, batch_size=self.batch_size).batches
        self.num_batches = len(self.batch_intervals)

        self.batches = [None]*self.num_batches
        self.set_weights(self.buffer)

    def set_weights(self, buffer):
        self.buffer = buffer
        for i in range(len(self.batch_intervals)):
            s, e = self.batch_intervals[i]
            self.batches[i] = buffer[s:e]

    def get_weights(self):
        for i in range(len(self.batch_intervals)):
            s, e = self.batch_intervals[i]
            self.buffer[s:e] = self.batches[i]
        return self.buffer

    def get_partition(self, i):
        return self.batches[i]

    def set_partition(self, i, buffer_batch):
        self.batches[i] = buffer_batch

    def add_partition(self, i, buffer_batch):
        self.batches[i] += buffer_batch


def add_results(filename, columns, results):
    print(*columns)
    if not os.path.isfile(filename):
        results = [columns] + results
    with open(filename, 'a') as file:
        for result in results:
            print(*result)
            line = ",".join(map(str, result))
            file.write(line+"\n")


class Worker(object):

    def __init__(self):
        self.workers = {}

    def ip(self):
        return ray.services.get_node_ip_address()

    def init(self, worker_index, num_workers, shape, dtype=np.float32):
        self.dtype = dtype
        self.size = int(shape[0])
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.weight_partition = WeightPartition(worker_index, num_workers, self.size, dtype=self.dtype)
        self.timeline = Timeline(worker_index)
        self.timeline.patch_ray()

    def reset_timeline(self):
        self.timeline.reset()

    def get_timeline(self):
        return self.timeline

    def add_remote_worker(self, index, worker):
        self.workers[index] = worker

    def sample(self, i=0):
        return self.weight_partition.buffer[i]

    def get_weights(self):
        return self.weight_partition.get_weights()

    def set_weights(self, weights):
        self.set_weights(weights)

    def node_address(self):
        return ray.services.get_node_ip_address()

    def ray_get(self, oid_bytes):
        plasma_id = ray.pyarrow.plasma.ObjectID(oid_bytes)
        ray.worker.global_worker.plasma_client.fetch([plasma_id])
        [raw_grads] = ray.worker.global_worker.plasma_client.get_buffers([plasma_id])
        return np.frombuffer(raw_grads, dtype=np.float32)

    def ray_put(self, oid_bytes, object):
        plasma_id = ray.pyarrow.plasma.ObjectID(oid_bytes)
        buff = ray.worker.global_worker.plasma_client.create(oid_bytes, object.nbytes)
        wrapper = np.frombuffer(buff, dtype=np.float32)
        np.copyto(wrapper, object)
        ray.worker.global_worker.plasma_client.seal(plasma_id)


class AllreduceClique(Worker):

    def execute(self, in_oid_bytes, out_oid_bytes, done_oid_bytes):
        raise NotImplementedError("Not done.")
        self.timeline.start("execute")
        self.in_oid_bytes = in_oid_bytes
        self.out_oid_bytes = out_oid_bytes
        self.done_oid_bytes = done_oid_bytes
        shard = self.ray_get(self.in_oid_bytes)
        self.weight_partition.set_weights(shard)
        self.iterate()
        self.timeline.end("execute")

    def iterate(self):
        self.timeline.start("iterate")
        # TODO(hme): Define these in constructor for clarity.
        self.partials_remaining = self.num_workers - 1
        self.sums_remaining = self.num_workers - 1
        self.start_time = time.time()
        self.broadcast_partial()
        self.timeline.end("iterate")

    def partial_done(self):
        self.timeline.start("partial_done")
        self.broadcast_sum()
        self.timeline.end("partial_done")

    def sums_done(self):
        self.timeline.start("sums_done")
        if self.partials_remaining == 0 and self.sums_remaining == 0:
            self.done()
        self.timeline.end("sums_done")

    def done(self):
        self.timeline.start("done")
        duration = time.time() - self.start_time
        self.timeline.end("done")

    def broadcast_partial(self):
        self.timeline.start("broadcast_partial")
        for index, worker in self.workers.items():
            if index != self.worker_index:
                self.timeline.start("broadcast_partial_" + str(index))
                batch_buffer = self.weight_partition.get_partition(index)
                oid = ray.put(batch_buffer)
                worker.receive_partial.remote(oid)
                self.timeline.end("broadcast_partial_" + str(index))
        self.timeline.end("broadcast_partial")

    def broadcast_sum(self):
        self.timeline.start("broadcast_sum")
        batch_buffer = self.weight_partition.get_partition(self.worker_index)
        oid = ray.put((self.worker_index, batch_buffer))
        for index, worker in self.workers.items():
            if index != self.worker_index:
                worker.receive_sum.remote(oid)
        self.timeline.end("broadcast_sum")

    def receive_partial(self, batch_buffer):
        self.timeline.start("receive_partial")
        self.weight_partition.add_partition(self.worker_index, batch_buffer)
        self.partials_remaining -= 1
        if self.partials_remaining == 0:
            self.partial_done()
        self.timeline.end("receive_partial")

    def receive_sum(self, obj):
        self.timeline.start("receive_sum")
        batch_index, batch_buffer = obj
        # assert batch_index != self.worker_index
        self.weight_partition.set_partition(batch_index, batch_buffer)
        self.sums_remaining -= 1
        if self.sums_remaining == 0:
            self.sums_done()
        self.timeline.end("receive_sum")


class AllreduceRing(Worker):

    def execute(self, in_oid_bytes, out_oid_bytes, done_oid_bytes):
        self.timeline.start("execute")
        self.in_oid_bytes = in_oid_bytes
        self.out_oid_bytes = out_oid_bytes
        self.done_oid_bytes = done_oid_bytes
        shard = self.ray_get(self.in_oid_bytes)
        shard.flags.writeable = True
        self.weight_partition.set_weights(shard)
        self.iterate()
        self.timeline.end("execute")

    def iterate(self):
        self.timeline.start("iterate")
        self.me = self.workers[self.worker_index]
        self.other = self.workers[(self.worker_index + 1) % self.num_workers]
        self.sr_remaining = self.num_workers - 1
        self.ag_remaining = self.num_workers - 1
        self.reset_iterator()
        self.execution_i = 0
        self.is_sr = True
        self.start_time = time.time()
        self.me.send.remote()
        self.timeline.end("iterate")

    def sr_done(self):
        self.timeline.event("sr_done")

    def ag_done(self):
        self.timeline.event("ag_done")
        assert self.sr_remaining == 0
        assert self.ag_remaining == 0
        self.done()

    def done(self):
        self.timeline.start("done")
        # write object
        self.ray_put(self.weight_partition.get_weights(), self.out_oid_bytes)
        duration = np.array([time.time() - self.start_time])
        self.ray_put(duration, self.done_oid_bytes)
        self.timeline.end("done")

    def reset_iterator(self):
        self.timeline.event("reset_iterator")
        self.iteration_index = (self.worker_index + 1) % self.num_workers

    def send_sr(self):
        self.timeline.start("send_sr")
        self.iteration_index = (self.iteration_index - 1) % self.num_workers
        batch_buffer = self.weight_partition.get_partition(self.iteration_index)
        self.timeline.end("send_sr")
        return self.iteration_index, batch_buffer

    def send_ag(self):
        self.timeline.start("send_ag")
        batch_buffer = self.weight_partition.get_partition(self.iteration_index)
        self.timeline.end("send_ag")
        return self.iteration_index, batch_buffer

    def receive_sr(self, obj):
        self.timeline.start("receive_sr")
        batch_index, batch_buffer = obj
        self.weight_partition.add_partition(batch_index, batch_buffer)
        self.sr_remaining -= 1
        if self.sr_remaining == 0:
            self.sr_done()
        self.timeline.end("receive_sr")

    def receive_ag(self, obj):
        self.timeline.start("receive_ag")
        batch_index, batch_buffer = obj
        batch_buffer.flags.writeable = True
        self.weight_partition.set_partition(batch_index, batch_buffer)
        self.iteration_index = batch_index
        self.ag_remaining -= 1
        if self.ag_remaining == 0:
            self.ag_done()
        self.timeline.end("receive_ag")

    def send(self):
        self.timeline.start("send")
        # print("send", self.worker_index, self.is_sr, self.iteration_index)
        if self.is_sr:
            oid = ray.put(self.send_sr())
        else:
            oid = ray.put(self.send_ag())
        self.other.receive.remote(oid)
        self.timeline.end("send")

    def receive(self, obj):
        self.timeline.start("receive")
        # print("receive", self.worker_index, self.is_sr, self.iteration_index)
        if self.is_sr:
            self.receive_sr(obj)
        else:
            self.receive_ag(obj)

        # update execution state
        self.execution_i += 1
        if self.is_sr and self.execution_i >= self.num_workers - 1:
            self.is_sr = False
            self.reset_iterator()
            # print("reset_iterator", self.worker_index, self.execution_i)

        if self.execution_i < 2*(self.num_workers - 1):
            # continue executing this allreduce
            # this must be synchronous, otherwise we risk receiving before sending the next iteration.
            self.send()
        self.timeline.end("receive")


def main(algorithm, check_results, num_workers, size, num_iterations, burn_k, redis_address):
    num_iterations += burn_k

    if algorithm == "ring":
        AlgoClass = AllreduceRing
    elif algorithm == "clique":
        AlgoClass = AllreduceClique

    ray.init(redis_address=args.redis_address, redirect_output=False, use_raylet=True)

    print("after init")

    # Create workers.
    workers = []
    for worker_index in range(num_workers):
        print("Using Actor", 'Actor' + str(worker_index+1))
        if redis_address is None:
            CurrWorker = ray.remote(AlgoClass)
        else:
            CurrWorker = ray.remote(resources={'Actor' + str(worker_index+1): 1})(AlgoClass)
        workers.append(CurrWorker.remote())
        workers[-1].init.remote(worker_index, num_workers, (size,))

    print("after starting workers")

    # Exchange actor handles.
    for i in range(num_workers):
        for j in range(num_workers):
            workers[i].add_remote_worker.remote(j, workers[j])

    print("after setting worker handles")

    # Ensure workers are assigned to unique nodes.
    node_ips = ray.get([worker.node_address.remote() for worker in workers])

    print("after getting ip addresses:", node_ips)

    for r in node_ips:
        print(r)
    print(len(set(node_ips)))
    if redis_address is not None:
      if not RAYLET_LOCAL_MULTINODE:
        assert len(set(node_ips)) == num_workers

    # generate oids
    oids = [None]*num_iterations
    for i in range(num_iterations):
        oids[i] = [None]*num_workers
        for j in range(num_workers):
            oids[i][j] = (ray.pyarrow.plasma.ObjectID(np.random.bytes(20)),
                          ray.pyarrow.plasma.ObjectID(np.random.bytes(20)),
                          ray.pyarrow.plasma.ObjectID(np.random.bytes(20)))

    # TODO(hme): Actually put objects in for in_oid.
    for i in range(num_iterations):
        done_oids = [None]*num_workers
        for j in range(num_workers):
            workers[j].execute.remote(oids[i][j][0], oids[i][j][1], oids[i][j][2])
            done_oids[j] = oids[i][j][2]

        done_plasma_ids = list(map(ray.pyarrow.plasma.ObjectID, done_oids))
        ray.worker.global_worker.plasma_client.fetch([done_plasma_ids])
        durations = ray.worker.global_worker.plasma_client.get_buffers([done_plasma_ids])
        print(np.max(durations))
        time.sleep(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmarks.')
    parser.add_argument('--algorithm', default="clique", type=str, help='The algorithm to use (ring or clique).')
    parser.add_argument('--check-results', action='store_true', help='Whether to check results.')
    parser.add_argument('--num-workers', default=4, type=int, help='The number of workers to use.')
    parser.add_argument('--size', default=25000000, type=int,
                        help='The number of 32bit floats to use.')
    parser.add_argument('--num-iterations', default=10, type=int,
                        help='The number of iterations.')
    parser.add_argument('--burn-k', default=1, type=int, help='Burn in first k iterations.')
    parser.add_argument('--redis-address', default=None, type=str,
                        help='The address of the redis server.')
    args = parser.parse_args()
    main(**vars(args))
