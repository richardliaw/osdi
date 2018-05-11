# README

`pip install clipper_admin`
`./resnetclip.py --runtime [ray] --num-sims [num] --iters [100]`
``./resnetclip.py --runtime [ray] --num-sims [num] --iters [500]`


 `for i in 1 2 4 8 16; do ./atariclip.py --runtime clipper --iters 50 --num-sims $i 2> logs.txt; done`
i
To generate latencies, I ran:
`./atariclip.py --runtime [ray/clipper/simple] --num-sims 1 --iters 50 [--big-model]`

for j in 1 2; do ./resnetclip.py --runtime ray --num-sims $j --iters 500 --depth 3 >> small_ray.txt; done && for j in 1 2; do ./resnetclip.py --runtime clipper --num-sims $j --iters 500 --depth 3 >> small_clip.txt; done
