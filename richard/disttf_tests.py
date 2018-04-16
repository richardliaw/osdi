import copy
import sys


def to_argv(kwargs):
    return ["--" + k + "=" + str(v) for k, v in kwargs.items()]


def get_private_ip(host):
    cmd = "hostname"
    res = ssh_cmd(["ssh", "-i NEW.pem", "ubuntu@{host}", "bash -c '{}'"])




def build_worker_cmd(kwargs):
    kwargs = copy.deepcopy(kwargs)
    kwargs.update({"job_name": "worker"})
    root = ["python", "tf_cnn_benchmarks.py"] + to_argv(kwargs)
    return " ".join(root)


########## PARAMETER SERVER

def build_ps_cmd(kwargs):
    kwargs = copy.deepcopy(kwargs)
    kwargs.update({"job_name": "ps"})
    root = ["CUDA_VISIBLE_DEVICES=", "python", "tf_cnn_benchmarks.py"
            ] + to_argv(kwargs)
    return " ".join(root)


def dist_replicated_cmd_builder(hosts, kwargs):
    def generate_host_flags():
        cmds = {}
        cmds["ps_hosts"] = ",".join([h + ":50000" for h in hosts])
        cmds["worker_hosts"] = ",".join([h + ":50001" for h in hosts])
        return cmds
    host_flags = generate_host_flags()
    kwargs.update(host_flags)
    cmds = {}
    for i, h in enumerate(hosts):
        hkwargs = copy.deepcopy(kwargs)
        hkwargs.update({"task_index": str(i)})
        print("#" * 10)
        print("Run the following commands on", h)
        print("#" * 10)
        print(build_ps_cmd(hkwargs) + " &")
        print("")
        print(build_worker_cmd(hkwargs))
        print("")
        cmds[h] = [build_ps_cmd(hkwargs), build_worker_cmd(hkwargs)]
    return cmds


########### ALL-REDUCE

def build_controller_cmd(kwargs):
    kwargs = copy.deepcopy(kwargs)
    kwargs.update({"job_name": "controller"})
    root = ["CUDA_VISIBLE_DEVICES=", "python", "tf_cnn_benchmarks.py"
            ] + to_argv(kwargs)
    return " ".join(root)


def dist_allreduce_cmd_builder(hosts, kwargs):
    def generate_host_flags():
        cmds = {}
        cmds["ps_hosts"] = ",".join([h + ":50000" for h in hosts])
        cmds["controller_host"] = [h + ":50001" for h in hosts][0]
        return cmds
    host_flags = generate_host_flags()
    kwargs.update(host_flags)
    cmds = {}
    for i, h in enumerate(hosts):
        hkwargs = copy.deepcopy(kwargs)
        hkwargs.update({"task_index": str(i)})
        print("#" * 10)
        print("Run the following commands on", h)
        print("#" * 10)
        ctrl_cmd = build_controller_cmd(hkwargs) if i == 0 else None
        print(ctrl_cmd)
        print("")
        print(build_worker_cmd(hkwargs))
        print("")
        cmds[h] = [ctrl_cmd, build_worker_cmd(hkwargs)]
    return cmds


if __name__ == '__main__':
    hosts = None
    with open("./hostfile") as f:
        hosts = [line.strip() for line in f if line.strip()]
    args = {
        'batch_size': '64',
        'local_parameter_device': 'gpu',
        'model': 'resnet101',
        'num_gpus': '8',
        'variable_update': 'distributed_replicated',
    }
    dist_replicated_cmd_builder(hosts, args)
