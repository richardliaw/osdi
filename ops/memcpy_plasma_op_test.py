import argparse
import time

import numpy as np

import ray
import ray.local_scheduler
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Whether to force ops to be placed on GPUs.")
args = parser.parse_args()

FORCE_DEVICE = '/gpu' if args.use_gpu else '/cpu'

memcpy_plasma_module = tf.load_op_library('./memcpy_plasma_op.so')
ray.init()
time.sleep(2)

plasma_store = ray.worker.global_worker.plasma_client.store_socket_name
plasma_manager = ray.worker.global_worker.plasma_client.manager_socket_name

object_id = np.random.bytes(20)
oid = ray.local_scheduler.ObjectID(object_id)

data = np.random.randn(3, 224, 224).astype(np.float32)

sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True))


def ToPlasma():
    return memcpy_plasma_module.tensor_to_plasma(
        data,
        object_id,
        plasma_store_socket_name=plasma_store,
        plasma_manager_socket_name=plasma_manager)


def FromPlasma():
    return memcpy_plasma_module.plasma_to_tensor(
        object_id,
        plasma_store_socket_name=plasma_store,
        plasma_manager_socket_name=plasma_manager)


with tf.device(FORCE_DEVICE):
    to_plasma = ToPlasma()
    from_plasma = FromPlasma()

    z = from_plasma + 1

sess.run(to_plasma)
print('Getting object...')
# NOTE(zongheng): currently it returns a flat 1D tensor.  So reshape manually.
out = sess.run(from_plasma).reshape(3, 224, 224)

sess.run(z)

assert np.array_equal(data, out), "Data not equal!"
print('OK: data all equal')
