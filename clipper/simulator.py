#!/usr/bin/env python 
import ray
import gym
import requests, json, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(36260, 50)
        self.fc2 = nn.Linear(50, 6)
        for layer in self.parameters():
            layer.requires_grad = False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 36260)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Simulator(object):
    def __init__(self, env):
        self._env = gym.make(env)
        self._init_state = self._env.reset()

    def onestep(self, arr, start=False):
        if start:
            return self._init_state
        state = self._env.step(arr)[0]
        return state

    def initial_state(self):
        return self._init_state


class ClipperRunner(Simulator):
    def __init__(self, env):
        super(ClipperRunner, self).__init__(env)
        self.start_clipper()

    def start_clipper(self):
        from clipper_admin import ClipperConnection, DockerContainerManager
        from clipper_admin.deployers import python as python_deployer
        self.clipper_conn = ClipperConnection(DockerContainerManager())
        try:
            self.clipper_conn.connect()
            self.clipper_conn.stop_all()
        except Exception:
            pass
        self.clipper_conn.start_clipper()
        self.clipper_conn.register_application(
            name="hello-world", input_type="doubles", 
            default_output="-1.0", slo_micros=100000)
        model = Model()
        def policy(xs):
            xs = [preprocess(x) for x in xs]
            res = model(convert_torch(xs))
            return from_torch(res)

        python_deployer.deploy_python_closure(
            self.clipper_conn, name="policy", version=1,
            input_type="doubles", func=policy)
        
        self.clipper_conn.link_model_to_app(
            app_name="hello-world", model_name="policy")
        self._headers = {"Content-type": "application/json"}

    def run(self, steps):
        state = self.initial_state()
        for i in range(steps):
            res = requests.post("http://localhost:1337/hello-world/predict", 
                                headers=headers, 
                                data=json.dumps({
                                    "input": list(state)
                                })
                  ).json()
            state = self.onestep(arr)


def eval():
    model = Model()
    sim = Simulator("Pong-v0")
    ac = None
    import time
    start = time.time()
    for i in range(50):
        xs = [sim.onestep(ac, i == 0)]
        xs = [preprocess(x) for x in xs]
        ac = model(convert_torch(xs))
        ac = from_torch(ac)[0].argmax()
    print("Took %0.4f sec..." % (time.time() - start))


if __name__ == "__main__":
    cr = ClipperRunner("Pong-v0")
    import ipdb; ipdb.set_trace()
    cr.run(500)
