#!/usr/bin/env python
import ray
import base64
import time
import gym
import requests, json, numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timers import TimerStat


import resnet


def preprocess(arr):
    H, W, C = arr.shape
    arr = arr.reshape(C, H, W)
    arr = arr.astype(np.float32)
    arr = (arr - 128) / 128
    return arr

def convert_torch(xs):
    xs = np.array(xs)
    return Variable(torch.from_numpy(xs))

def from_torch(xs):
    return xs.data.numpy()

class ModelBig(nn.Module):
    def __init__(self):
        super(ModelBig, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(400, 6)
        for layer in self.parameters():
            layer.requires_grad = False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.max_pool2d(x, 3, 3)
        x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)


class ModelSimple(nn.Module):
    def __init__(self):
        super(ModelSimple, self).__init__()
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


def evaluate_model(model, xs):
    """
    Args:
        xs: (N, shape)
    """
    res = model(convert_torch(xs))
    return from_torch(res).argmax(axis=1)

def get_model(use_big=False):
    if use_big:
        return ModelBig()
    else:
        return ModelSimple()

class Simulator(object):
    def __init__(self, args):
        self._env = gym.make(args.env)
        _state = self._env.reset()
        self._init_state = np.array([preprocess(_state) for i in range(args.batch)])

    def onestep(self, arr, start=False):
        self._init_state += 0.001
        # if start:
        #     return self._init_state
        # state = self._env.step(arr)[0]

        return self._init_state

    def initial_state(self):
        return self._init_state


class Clip(object):
    def __init__(self, shape, use_big=False):
        from clipper_admin import ClipperConnection, DockerContainerManager
        from clipper_admin.deployers import python as python_deployer
        from clipper_admin.deployers import pytorch as pytorch_deployer
        self.clipper_conn = ClipperConnection(DockerContainerManager())
        try:
            self.clipper_conn.connect()
            self.clipper_conn.stop_all()
        except Exception:
            pass
        self.clipper_conn.start_clipper()
        self.clipper_conn.register_application(
            name="hello-world", input_type="strings",
            default_output="-1.0", slo_micros=10**8)
        ptmodel = get_model(use_big)
        def policy(model, x):
            print(len(x))
            batch = (len(x))
            arr = []
            for j in x:
                print(type(j), len(j))
                res = np.frombuffer(base64.decodestring(j), dtype=np.float32)
                print(res.shape)
                arr += [res]
            x = np.array(arr)
            x = x.reshape((-1,)  + shape[1:])
            print("new shape", x.shape)
            return evaluate_model(model, x).reshape((batch, shape[0]))
        pytorch_deployer.deploy_pytorch_model(
            self.clipper_conn, name="policy", version=1,
            input_type="strings", func=policy, pytorch_model=ptmodel)

        self.clipper_conn.link_model_to_app(
            app_name="hello-world", model_name="policy")


# class PolicyActor(object):
#     def __init__(self):
#         self.ptmodel = Model()
# 
#     def query(self, state):
#         state = [state]
#         return evaluate_model(self.ptmodel, state)


class ClipperRunner(Simulator):
    def __init__(self, args):
        super(ClipperRunner, self).__init__(args)
        self.shape = self.initial_state().shape
        self._headers = {"Content-type": "application/json"}

    def run(self, steps):
        state = self.initial_state()
        serialize_timer = TimerStat()
        step_timer = TimerStat()
        for i in range(steps):
            assert len(state.shape) == 4
            with step_timer:
                with serialize_timer:
                    s = base64.b64encode(state)
                    data = json.dumps({"input": s})
                res = requests.post(
                    "http://localhost:1337/hello-world/predict",
                    headers=self._headers,
                    data=data).json()
                out = res['output']
                state = self.onestep(out)
        print("Serialize", serialize_timer.mean)
        print("Step", step_timer.mean)


# class RayRunner(Simulator):
#     def __init__(self, env):
#         super(RayRunner, self).__init__(env)
#         self.shape = self.initial_state().shape
#         self.timers = {"query": TimerStat(), "step": TimerStat()}

#     def run(self, steps, policy_actor):
#         state = self.initial_state()
#         for i in range(steps):
#             with self.timers["query"]:
#                 out = ray.get(policy_actor.query.remote(state))

#             with self.timers["step"]:
#                 state = self.onestep(out)

#     def stats(self):
#         return {k: v.mean for k, v in self.timers.items()}


def eval_ray_batch(args):
    model = get_model(args.use_big)
    RemoteSimulator = ray.remote(Simulator)
    simulators = [RemoteSimulator.remote(args) for i in range(args.num_sims)]
    ac = [None for i in range(args.num_sims)]
    start = time.time()
    init_shape = ray.get(simulators[0].initial_state.remote()).shape
    remaining = {sim.onestep.remote(a, i == 0): sim for a, sim in zip(ac, simulators)}
    counter = {sim: 0 for sim in simulators}
    fwd = TimerStat()
    while any(v < args.iters for v in counter.values()):
        # TODO: consider evaluating as ray.wait
        [data_fut], _ = ray.wait(list(remaining))
        xs = ray.get(data_fut)
        sim = remaining.pop(data_fut)
        counter[sim] += 1

        with fwd:
            ac = evaluate_model(model, xs)
        print(xs.shape, ac.shape)
        if counter[sim] < args.iters:
            remaining[sim.onestep.remote(ac[0], i == 0)] = sim
    print("Took %0.4f sec..." % (time.time() - start))
    print(fwd.mean)


def eval_simple(args):
    model = get_model(args.use_big)
    sim = Simulator(args)
    fwd = TimerStat()
    start = time.time()
    ac = [None]
    for i in range(args.iters):
        xs = sim.onestep(ac[0], i == 0)
        with fwd:
            ac = evaluate_model(model, xs)
    print("Took %0.4f sec..." % (time.time() - start))
    print(fwd.mean, "Avg Fwd pass..")

# def eval_ray(args):
#     RemoteRayRunner = ray.remote(RayRunner)
#     simulators = [RemoteRayRunner.remote(args) for i in range(args.num_sims)]
#     RemotePolicy = ray.remote(PolicyActor)
#     p = RemotePolicy.remote()
#     start = time.time()
#     ray.get([sim.run.remote(args.iters, p) for sim in simulators])
#     print("Took %0.4f sec..." % (time.time() - start))
#     stats = ray.get(simulators[0].stats.remote())
#     print(stats)


def eval_clipper(args):
    RemoteClipperRunner = ray.remote(ClipperRunner)
    simulators = [RemoteClipperRunner.remote(args) for i in range(args.num_sims)]
    c = Clip(ray.get(simulators[0].initial_state.remote()).shape, args.use_big)
    start = time.time()
    ray.get([sim.run.remote(args.iters) for sim in simulators])
    print("Took %0.4f sec..." % (time.time() - start))


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
parser.add_argument("--use-big", action="store_true",
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
