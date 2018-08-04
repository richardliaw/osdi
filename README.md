# OSDI 2018 (Distributed Training Experiments)

For Horovod, Distributed TF, and Ray experiments


I always forget the experiment command used: 

./resnet101-ray-sgd.py --verbose --devices-per-actor=8 --plasma-op --max-bytes=10000000 --ps --roundrobin_ps --warmup --ps-spinwait --num-actors=7 --cluster --debug_ps
