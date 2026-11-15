"""Does a resolvable gap between the two interlocking regions change the result?"""
import sys
sys.path.insert(0, '.')
import numpy as np
from shapely.geometry import box
from sklearn.metrics import adjusted_rand_score
import adapters as ad
import scene_gen as sg

def build_c(gap):
    disc = sg.disk(46, 50, 34)
    bite = box(46, 22, 92, 78)
    body = disc.difference(bite)
    fill = bite.intersection(disc.buffer(1.0)).buffer(-gap)
    return [body, fill]

def build_u(gap):
    hole = box(30, 40, 70, 92)
    body = box(26, 16, 86, 80).difference(hole)
    fill = box(30, 40, 70, 80).buffer(-gap)
    return [body, fill]

rng = np.random.default_rng(9001)
for name, fn in (('C', build_c), ('U', build_u)):
    for gap in (1.8, 3.0, 4.5):
        polys = fn(gap)
        areas = np.array([p.area for p in polys])
        counts = np.clip(np.round(300 * areas / areas.min()), 300, 500).astype(int)
        Xs, ys = [], []
        for i, (p, n) in enumerate(zip(polys, counts), 1):
            pts = sg.sample_in(p, int(n), rng)
            Xs.append(pts); ys.append(np.full(len(pts), i))
        X = np.vstack(Xs) + rng.normal(0, 0.15, (sum(counts), 2))
        y = np.concatenate(ys)
        out = []
        for g in (1.0, 2.0, 3.0):
            lab, _ = ad.run('IZGC-Full', X, 2, dict(grid_size=g, R=30, min_size=5, buffer=0),
                            scene=dict(polys=polys, barrier_line=None), polygonise=False)
            k = len(set(int(v) for v in lab) - {-1})
            out.append(f'g={g}: ARI={adjusted_rand_score(y, lab):.2f} k={k}')
        print(f'{name} gap={gap}  counts={counts}  ' + '  '.join(out), flush=True)
