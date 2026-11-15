"""Run a single configuration; used with a hard timeout to spot pathological cases."""
import sys, time
sys.path.insert(0, '.')
import adapters as ad, scene_gen as sg

name = sys.argv[1]
params = eval(sys.argv[2])
scene = sg.build(sys.argv[3] if len(sys.argv) > 3 else 'nonconvex_u', 9001)
t = time.time()
lab, _ = ad.run(name, scene['X'], scene['k'], params, scene=scene, polygonise=False)
k = len(set(int(v) for v in lab) - {-1})
print(f"{name} {params} -> {time.time() - t:.2f}s  k={k}")
