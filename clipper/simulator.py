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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(36260, 50)
        self.fc2 = nn.Linear(50, 10)

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

    def onestep(self, arr):
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
        def feature_sum(xs):
            xs = [preprocess(x) for x in xs]
            res = model(convert_torch(xs))
            return res

        python_deployer.deploy_python_closure(
            self.clipper_conn, name="sum-model", version=1,
            input_type="doubles", func=feature_sum)
        
        self.clipper_conn.link_model_to_app(
            app_name="hello-world", model_name="sum-model")
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
            arr = 0
            state = self.onestep(arr)


if __name__ == "__main__":
    model = Model()
    sim = Simulator("Pong-v0")
    xs = [sim.initial_state()]
    xs = [preprocess(x) for x in xs]
    import ipdb; ipdb.set_trace()
    res = model(convert_torch(xs))
