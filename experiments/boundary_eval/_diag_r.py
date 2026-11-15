"""How sensitive is IZGC to the search radius on these scenes?"""
import sys
sys.path.insert(0, '.')
import numpy as np
import adapters as ad
import scene_gen as sg
from sklearn.metrics import adjusted_rand_score

for nm in ('nonconvex_c', 'gap_4', 'density_8'):
    s = sg.build(nm, 9001)
    print(f'--- {nm}  n={len(s["X"])}  k={s["k"]}')
    for R in (5, 10, 15, 20, 30):
        row = []
        for g in (1.0, 2.0, 3.0, 4.0):
            lab, _ = ad.run('IZGC-Full', s['X'], s['k'],
                            dict(grid_size=g, R=R, min_size=5, buffer=0),
                            scene=s, polygonise=False)
            k = len(set(int(v) for v in lab) - {-1})
            row.append(f'g={g}: ARI={adjusted_rand_score(s["y"], lab):.2f} k={k}')
        print(f'  R={R:2d}  ' + '  '.join(row))
