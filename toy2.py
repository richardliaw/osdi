from __future__ import print_function

import tensorflow as tf
import ray

@ray.remote(num_gpus=8)
class Actor(object):
    def hello(self):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        # Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # Runs the op.
        return (sess.run(c))

ray.init(redis_address="localhost:6379")

actors  = []
for a in range(2):
    actors += [Actor.remote()]
    import time
    time.sleep(1)

print(ray.get([a.hello.remote() for a in actors]))
