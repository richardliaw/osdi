#!/usr/bin/env python
import ray
import requests, json
import numpy as np
import time
from timers import TimerStat
import resnet
import torch


class AGSimulator(object):
    def __init__(self, boardsize=19, batch_size=64):
        self.batch_size = batch_size
        self.boardsize = boardsize
        self.action_size = self.boardsize**2 + 1
        state_size = (self.batch_size, self.boardsize, self.boardsize, 3)
        self._init_state = np.random.normal(size=state_size).astype(np.float32)
        self._init_mask = np.zeros((self.batch_size, self.action_size), dtype=np.float32)

    def onestep(self, arr):
        xs = np.random.normal(size=(self.batch_size, self.boardsize, self.boardsize, 3)).astype(np.float32)
        masks = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
        return xs, masks

    def initial_state(self):
        return self._init_state, self._init_mask


# @ray.remote(num_gpus=1)
class ResNetModel(object):
    def __init__(self, batch_size=64):
        self.boardsize = 19
        self.game_name = 'go'
        self.action_size = self.boardsize**2 + (1 if self.game_name == 'go' else 0)
        self.batch_size = batch_size
        self.params = {
            'boardsize': self.boardsize,
            'batchn': True,
            'channels': 256,
            'stack_depth': 19,  # Should be 19 or 39
            'nonlinearity': 'relu',
            'mask': True,  # Should be True
            'policy_head': 'az_stone',
            'policy_head_params': {'boardsize': self.boardsize, 'actionsize': self.action_size},
            'value_head': 'az',
            'value_head_params': {'boardsize': self.boardsize, 'fc_width': 256},
            'value_nonl': 'tanh',
            'l2_strength': 10**-4,
            'optimizer_name': 'momentum',
            'optimizer_params': {'momentum': 0.9},
            'lr_boundaries': [int(2e5), int(4e5), int(6e5)],
            'lr_values': [0.2, 0.02, 0.002, 0.0002]
        }

        self.not_estimator = resnet.NotEstimator(resnet.resnet_model_fn, self.params)
        self.not_estimator.initialize_graph()

        # self.clients = [AlphaGoClient.remote(self.boardsize, self.action_size, self.batch_size)
        #                 for _ in range(self.num_clients)]

    # def run(self, num_clients, num_iters):
    #     num_clients = num_clients
    #     remote_agclient = ray.remote(alpha_go_client)
    #     remaining_ids = [remote_agclient.remote(
    #                         None, self.boardsize, self.action_size, self.batch_size)
    #                         for i in range(num_clients)]
    #     for _ in range(num_iters):
    #         [ready_id], remaining_ids = ray.wait(remaining_ids)
    #         xs, masks = ray.get(ready_id)
    #         values = self.not_estimator.predict(xs, masks)['value']
    #         remaining_ids.append(remote_agclient.remote(values, self.boardsize, self.action_size, self.batch_size))
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(100800, 6)
        # self.fc2 = nn.Linear(50, 6)
        for layer in self.parameters():
            layer.requires_grad = False

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = F.max_pool2d(x, 3, 3)
        #x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 100800)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Clip(object):
    def __init__(self):
        from clipper_admin import ClipperConnection, DockerContainerManager
        #from clipper_admin.deployers import python as python_deployer
        from clipper_admin.deployers import pytorch as pt_deployer
        self.clipper_conn = ClipperConnection(DockerContainerManager())
        try:
            self.clipper_conn.connect()
            self.clipper_conn.stop_all()
        except Exception:
            pass
        self.clipper_conn.start_clipper()
        self.clipper_conn.register_application(
            name="hello-world", input_type="floats",
            default_output="-1.0", slo_micros=10**8)
        model = Model()
        def policy(ptmodel, x):
            print(len(x))
            batch = (len(x))
            arr = []
            for j in x:
                print(type(j), len(j))
                res = np.frombuffer(base64.decodestring(j), dtype=np.float32)
                print(res.shape)
            for i in x:
                time.sleep(0.053)
            return [np.random.rand(64).astype(np.float32) for i in range(batch)]
        pt_deployer.deploy_pytorch_model(
            self.clipper_conn, name="policy", version=1,
            input_type="floats", func=policy, pytorch_model=model)

        self.clipper_conn.link_model_to_app(
            app_name="hello-world", model_name="policy")


class ClipperRunner(AGSimulator):
    def __init__(self):
        super(ClipperRunner, self).__init__()
        self._headers = {"Content-type": "application/json"}

    def run(self, steps):
        state = self.initial_state()
        serialize_timer = TimerStat()
        step_timer = TimerStat()
        for i in range(steps):
            with step_timer:
                with serialize_timer:
                    s = [base64.b64encode(xs), base64.b64encode(masks)]
                    data = json.dumps({"input": s})
                res = requests.post(
                    "http://localhost:1337/hello-world/predict",
                    headers=self._headers,
                    data=data).json()
                out = res['output']
                state = self.onestep(out)
        print("Serialize", serialize_timer.mean)
        print("Step", step_timer.mean)


def eval_ray_batch(args):
    estimator = ResNetModel().not_estimator
    RemoteAGSimulator = ray.remote(AGSimulator)
    simulators = [RemoteAGSimulator.remote() for i in range(args.num_sims)]
    ac = [None for i in range(args.num_sims)]
    init_shape = ray.get(simulators[0].initial_state.remote()).shape
    remaining = {sim.onestep.remote(a): sim for a, sim in zip(ac, simulators)}
    counter = {sim: 0 for sim in simulators}
    timers = {k: TimerStat() for k in ["fwd", "wait", "get",  "step"]}
    start = time.time()
    while any(v < args.iters for v in counter.values()):
        # TODO: consider evaluating as ray.wait
        with timers["step"]:
            with timers["wait"]:
                [data_fut], _ = ray.wait(list(remaining))
            with timers["get"]:
                xs, masks = ray.get(data_fut)
            sim = remaining.pop(data_fut)
            counter[sim] += 1

            with timers["fwd"]:
                values = estimator.predict(xs, masks)['value']
            if counter[sim] < args.iters:
                remaining[sim.onestep.remote(values)] = sim
    print("Took %0.4f sec..." % (time.time() - start))
    print(xs.shape)
    print("\n".join(["%s: %0.5f" % (k, t.mean) for k, t in timers.items()]))


def eval_clipper(args):
    RemoteClipperRunner = ray.remote(ClipperRunner)
    simulators = [RemoteClipperRunner.remote() for i in range(args.num_sims)]
    c = Clip()
    start = time.time()
    ray.get([sim.run.remote(args.iters) for sim in simulators])
    print("Took %0.4f sec..." % (time.time() - start))


def eval_simple(args):
    model = get_model(args.model)
    sim = Simulator(args)
    fwd = TimerStat()
    start = time.time()
    ac = [None]
    for i in range(args.iters):
        xs = sim.onestep(ac[0], i == 0)
        with fwd:
            values = estimator.predict(xs, masks)['value']
    print("Took %f sec..." % (time.time() - start))
    print(fwd.mean, "Avg Fwd pass..")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=str, choices=["ray", "clipper", "simple"],
    help="Choose between Ray or Clipper")
parser.add_argument("--env", type=str, default="Pong-v0",
    help="Env Keyword for starting a simulator")
parser.add_argument("--batch", type=int, default=1,
    help="Size of data")
parser.add_argument("--num-sims", type=int, default=1,
    help="Number of simultaneous simulations to evaluate")
parser.add_argument("--iters", type=int, default=500,
    help="Number of steps per sim to evaluate")
parser.add_argument("--model", type=str, default="simple",
    help="Use a bigger CNN model.")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.runtime == "ray":
        import ray
        ray.init()
        eval_ray_batch(args)
    elif args.runtime == "clipper":
        import ray
        ray.init()
        eval_clipper(args)
    elif args.runtime == "simple":
        eval_simple(args)
