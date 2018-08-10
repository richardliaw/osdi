# OSDI 2018 (Distributed Training Experiments)

For Horovod, Distributed TF, and Ray experiments

## As of Aug 9, 2018, to run the SGD experiments:

use `./resnet101-ray-sgd.py --devices-per-actor=8 --plasma-op --max-bytes=100000000 --ps --warmup --ps-spinwait --num-actors=2 --cluster --debug_ps`

And use `sgd.yaml` to launch the cluster. You may need to reset the conda env when you finally link up with the cluster.

The Ray commit used here is https://github.com/pcmoritz/ray-1/commit/136568f07c59f353dbfdac8e13baf3ba6839df98.
