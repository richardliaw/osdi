import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')

import ray
ray.init()
plasma_store = ray.worker.global_worker.plasma_client.store_socket_name
plasma_manager = ray.worker.global_worker.plasma_client.manager_socket_name

sess = tf.InteractiveSession()
memcpy = zero_out_module.memcpy_plasma(
    tf.random_normal([5, 5]),
    "myobjectid",
    plasma_store_socket_name=plasma_store,
    plasma_manager_socket_name=plasma_manager)
print(sess.run(memcpy))
