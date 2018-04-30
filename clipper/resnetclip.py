#!/usr/bin/env python
import ray
import requests, json, numpy as np
import time
from timers import TimerStat
import resnet


class AGSimulator(object):
    def __init__(self, boardsize, action_size, batch_size):
        self.batch_size = batch_size
        self.boardsize = boardsize
        self.action_size = actionsize

    def onestep(self, arr):
        xs = np.random.normal(size=(self.batch_size, self.boardsize, self.boardsize, 3)).astype(np.float32)
        masks = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
        return xs, masks

    def initial_state(self):
        return self._init_state


# @ray.remote(num_gpus=1)
class ResNetModel(object):
    def __init__(self, batch_size):
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

    def run(self, num_clients, num_iters):
        num_clients = num_clients
        remote_agclient = ray.remote(alpha_go_client)
        remaining_ids = [remote_agclient.remote(
                            None, self.boardsize, self.action_size, self.batch_size)
                            for i in range(num_clients)]
        for _ in range(num_iters):
            [ready_id], remaining_ids = ray.wait(remaining_ids)
            xs, masks = ray.get(ready_id)
            values = self.not_estimator.predict(xs, masks)['value']
            remaining_ids.append(remote_agclient.remote(values, self.boardsize, self.action_size, self.batch_size))


class ClipTF(object):
    def __init__(self, shape):
        from clipper_admin import ClipperConnection, DockerContainerManager
        # from clipper_admin.deployers import python as python_deployer
        from clipper_admin.deployers import tensorflow as tf_deployer
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
        resnet = ResNetModel()
        def policy(model, x):
            batch = (len(x))
            x = np.array(x)
            x = x.reshape((batch * shape[0],)  + shape[1:])
            return model.predict(xs, masks)['value']
        tf_deployer.deploy_tensorflow_model(
            self.clipper_conn, name="policy", version=1,
            input_type="floats", func=policy,
            tf_sess_or_saved_model_path=resnet.not_estimator.sess)

        self.clipper_conn.link_model_to_app(
            app_name="hello-world", model_name="policy")


class ClipperRunner(Simulator):
    def __init__(self, env):
        super(ClipperRunner, self).__init__(env)
        self.shape = self.initial_state().shape
        self._headers = {"Content-type": "application/json"}

    def run(self, steps):
        state = self.initial_state()
        serialize_timer = TimerStat()
        for i in range(steps):
            assert len(state.shape) == 4
            with serialize_timer:
                s = list(state.astype(float).flatten())
                data = json.dumps({"input": s})
            res = requests.post(
                "http://localhost:1337/hello-world/predict",
                headers=self._headers,
                data=data).json()
            print(res)
            out = res['output']
            state = self.onestep(out)
        print("Mean", serialize_timer.mean)


def eval_ray_batch(args):
    model = Model()
    RemoteSimulator = ray.remote(Simulator)
    simulators = [RemoteSimulator.remote(args.env) for i in range(args.num_sims)]
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

def eval_clipper(args):
    RemoteClipperRunner = ray.remote(ClipperRunner)
    simulators = [RemoteClipperRunner.remote(args.env) for i in range(args.num_sims)]
    c = Clip(ray.get(simulators[0].initial_state.remote()).shape)
    start = time.time()
    ray.get([sim.run.remote(args.iters) for sim in simulators])
    print("Took %0.4f sec..." % (time.time() - start))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=str, choices=["ray", "clipper", "raybatch"],
    help="Choose between Ray or Clipper")
parser.add_argument("--env", type=str, default="Pong-v0",
    help="Env Keyword for starting a simulator")
parser.add_argument("--num-sims", type=int, default=1,
    help="Number of simultaneous simulations to evaluate")
parser.add_argument("--iters", type=int, default=500,
    help="Number of steps per sim to evaluate")


def ray_main():
    ray.init()
    num_clients = 2
    num_iters = 2
    batch_size = 64
    server = ResNetModel(num_clients, batch_size)
    start = time.time()
    server.run(num_iters)
    duration = time.time() - start
    num_processed = batch_size * num_iters
    print('Processed {} boards per second.'.format(num_processed / duration))


if __name__ == "__main__":
    args = parser.parse_args()
    import ray
    ray.init()
    if args.runtime == "ray":
        eval_ray_batch(args)
    elif args.runtime == "clipper":
        eval_clipper(args)
