# Launching Distributed TF Experiments

`ray create_or_update tfbench.yaml`

You may need to modify the file sources to point to your own dir.
Then, ssh onto the machine and run 
```
source activate tensorflow_p27
python launch.py [--help]
```

Notes: 
 - distributed replicated doesn't work for 16 machines
 - distributed_all_reduce ... doesn't work at all.
