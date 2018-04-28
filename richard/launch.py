import ray
import subprocess
import os
from disttf_tests import dist_replicated_cmd_builder, dist_allreduce_cmd_builder
import pipes
import time


@ray.remote(num_gpus=8)
class Runner(object):
    def __init__(self):
        self.others = []

    def run_cmd(self, cmd, blocking=False, env_vars={}):
        changedir = "cd benchmarks/scripts/tf_cnn_benchmarks/"
        activate = "source activate tensorflow_p27"
        tf_command = "{}".format(pipes.quote(" && ".join([activate, changedir, cmd])))
        if blocking:
            res = subprocess.check_output(["bash -c {}".format(tf_command)],
                                          shell=True,
                                          env=os.environ.copy().update(env_vars))
            return res
        else:
            res = subprocess.Popen(["bash -c {}".format(tf_command)],
                                   shell=True,
                                   env=os.environ.copy().update(env_vars))
            self.others += [res]

    def get_hostname(self):
        print("getting hostname")
        return ray.services.get_node_ip_address()

    def cleanup(self):
        return subprocess.check_output(["bash", "cleanup.sh"])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--num-ps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--var-update", type=str, default='distributed_replicated')
    parser.add_argument("--remote", type=bool, default=True)
    parsed_args = parser.parse_args()

    if parsed_args.remote:
        ray.init(redis_address=(ray.services.get_node_ip_address() + ":6379"), redirect_output=False)
    else:
        ray.init(num_gpus=parsed_args.nodes* 8)
    runners = [Runner.remote() for i in range(parsed_args.nodes)]
    hosts = ray.get([r.get_hostname.remote() for r in runners])
    ray.get([r.cleanup.remote() for r in runners])
    host_to_runner = dict(zip(hosts, runners))
    tf_args = {
        'batch_size': str(parsed_args.batch),
        'model': 'resnet101',
        'num_gpus': str(parsed_args.gpus_per_node),
        'variable_update': parsed_args.var_update,
    }
    if parsed_args.var_update == "distributed_replicated":
        host_cmds = dist_replicated_cmd_builder(hosts, tf_args, parsed_args.num_ps)
        for host, (ps_cmd, worker_cmd) in host_cmds.items():
            runner = host_to_runner[host]
            if not ps_cmd: 
                continue
            print("Launching ps for {}...".format(host))
            runner.run_cmd.remote(ps_cmd + " > ~/ps.out 2>&1", env_vars={"CUDA_VISIBLE_DEVICES": ""})
            print(ps_cmd)

        time.sleep(1)

        for host, (ps_cmd, worker_cmd) in host_cmds.items():
            runner = host_to_runner[host]
            print("Launching worker for {}...".format(host))
            final = runner.run_cmd.remote(worker_cmd + " > ~/worker.out 2>&1",
                                          blocking=True, env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
            print(worker_cmd)
    elif parsed_args.var_update == "distributed_all_reduce":
        all_reduce_args = {"all_reduce_spec": "nccl/xring"}
        all_reduce_args.update(tf_args)
        host_cmds = dist_allreduce_cmd_builder(hosts, all_reduce_args)
        for host, (ctrl_cmd, worker_cmd) in host_cmds.items():
            runner = host_to_runner[host]
            if ctrl_cmd:
                print("Launching ctrl for {}...".format(host))
                runner.run_cmd.remote(ctrl_cmd + " > ~/ctrl.out 2>&1", env_vars={"CUDA_VISIBLE_DEVICES": ""})
                print(ctrl_cmd)
                time.sleep(10)

            print("Launching worker for {}...".format(host))
            final = runner.run_cmd.remote(worker_cmd + " > ~/worker.out 2>&1",
                                          blocking=True, env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
            print(worker_cmd)

    result = ray.get(final)
    print(result)





