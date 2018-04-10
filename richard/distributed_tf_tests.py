import copy
import sys


def to_argv(kwargs):
    return ["--" + k + "=" + str(v) for k, v in kwargs.items()]


def generate_host_flags(hosts):
    cmds = {}
    cmds["ps_hosts"] = ",".join([h + ":50000" for h in hosts])
    cmds["worker_hosts"] = ",".join([h + ":50001" for h in hosts])
    return cmds


def build_worker_cmd(kwargs):
    kwargs = copy.deepcopy(kwargs)
    kwargs.update({"job_name": "worker"})
    root = ["python", "tf_cnn_benchmarks.py"] + to_argv(kwargs)
    return " ".join(root)


def build_ps_cmd(kwargs):
    kwargs = copy.deepcopy(kwargs)
    kwargs.update({"job_name": "ps"})
    root = ["CUDA_VISIBLE_DEVICES=", "python", "tf_cnn_benchmarks.py"
            ] + to_argv(kwargs)
    return " ".join(root)


def cmd_builder(hosts, kwargs):
    host_flags = generate_host_flags(hosts)
    kwargs.update(host_flags)
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


if __name__ == '__main__':
    hosts = None
    with open(sys.argv[-1]) as f:
        hosts = [line.strip() for line in f]
    args = {
        'batch_size': '64',
        'local_parameter_device': 'gpu',
        'model': 'resnet101',
        'num_gpus': '8',
        'variable_update': 'distributed_replicated',
    }
    cmd_builder(hosts, args)
