import ray
import subprocess
import os
from disttf_tests import dist_replicated_cmd_builder
import pipes

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


if __name__ == '__main__':
    import os
    try:
        os.system("ps -aux | grep tf | awk -F  " " '/1/ {print $2}' | xargs kill -9")
    except Exception as e:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--update", type=str, default='distributed_replicated')
    parser.add_argument("--remote", action='store_true')
    args = parser.parse_args()

    if args.remote:
        ray.init(redis_address=(ray.services.get_node_ip_address() + ":6379"), redirect_output=False)
    else:
        ray.init(num_gpus=args.num)
    runners = [Runner.remote() for i in range(args.num)]
    hosts = ray.get([r.get_hostname.remote() for r in runners])
    host_to_runner = dict(zip(hosts, runners))
    args = {
        'batch_size': str(args.batch),
        'local_parameter_device': 'gpu',
        'model': 'resnet101',
        'num_gpus': '8',
        'variable_update': args.update,
    }
    if args.update == "distributed_replicated":
        host_cmds = dist_replicated_cmd_builder(hosts, args)
        for host, (ps_cmd, worker_cmd) in host_cmds.items():
            runner = host_to_runner[host]
            print("Launching ps for {}...".format(host))
            runner.run_cmd.remote(ps_cmd + " > ~/ps.out 2>&1", env_vars={"CUDA_VISIBLE_DEVICES": ""})
            print(ps_cmd)

        import time
        time.sleep(10)

        for host, (ps_cmd, worker_cmd) in host_cmds.items():
            runner = host_to_runner[host]
            print("Launching worker for {}...".format(host))
            final = runner.run_cmd.remote(worker_cmd + " > ~/worker.out 2>&1",
                                          blocking=True, env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
            print(worker_cmd)
    elif args.update == "distributed_all_reduce":
        host_cmds = dist_allreduce_cmd_builder(hosts, args)
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





