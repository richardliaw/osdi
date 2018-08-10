from resnet import ParameterServer

import ray; ray.shutdown()
ray.init()
import numpy as np
put_oid_bin = np.random.bytes(20)
get_oid_bin = np.random.bytes(20)

random_array = np.random.rand(5,5)
print random_array

put_oid = ray.pyarrow.plasma.ObjectID(put_oid_bin)
ray.worker.global_worker.plasma_client.put(random_array.flatten(), object_id=put_oid)

ps = ParameterServer(1, 1)
ps.initialize(25)
ps.add(put_oid_bin)
ps.get(get_oid_bin)

get_oid = ray.pyarrow.plasma.ObjectID(get_oid_bin)
print ray.worker.global_worker.plasma_client.get(get_oid)
