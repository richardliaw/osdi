from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer
import requests, json, numpy as np

restart = False
clipper_conn = ClipperConnection(DockerContainerManager())
if restart:
    clipper_conn.start_clipper()
else:
    clipper_conn.connect()
import ipdb; ipdb.set_trace()
if "hello-world" not in clipper_conn.get_all_apps():
    clipper_conn.register_application(
        name="hello-world", input_type="doubles", 
        default_output="-1.0", slo_micros=100000)
    
    def feature_sum(xs):
        return [str(sum(x)) for x in xs]
    python_deployer.deploy_python_closure(clipper_conn, name="sum-model", version=1,
                                          input_type="doubles", func=feature_sum)
    
    clipper_conn.link_model_to_app(app_name="hello-world", model_name="sum-model")
else:
    pass
headers = {"Content-type": "application/json"}
res = requests.post("http://localhost:1337/hello-world/predict", headers=headers, 
              data=json.dumps({"input": list(np.random.random(10))})).json()
print(res)
