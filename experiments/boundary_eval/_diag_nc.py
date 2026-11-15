"""Check that the non-convex scenes separate IZGC from a linear-boundary method."""
import sys
sys.path.insert(0, '.')
import numpy as np
from shapely.geometry import box
from sklearn.metrics import adjusted_rand_score
import adapters as ad
import scene_gen as sg

def build(name):
    if name == 'c':
        disc = sg.disk(46, 50, 34)
        bite = box(46, 22, 92, 78)
        body = disc.difference(bite)
        fill = bite.intersection(disc.buffer(1.0)).buffer(-4.5)
    else:
        hole = box(28, 38, 72, 92)
        body = box(26, 16, 86, 80).difference(hole)
        fill = box(28, 38, 72, 80).buffer(-4.5)
    return [body, fill]

rng = np.random.default_rng(9001)
for nm in ('c', 'u'):
    polys = build(nm)
    areas = np.array([p.area for p in polys])
    counts = np.clip(np.round(300 * areas / areas.min()), 300, 500).astype(int)
    Xs, ys = [], []
    for i, (p, n) in enumerate(zip(polys, counts), 1):
        pts = sg.sample_in(p, int(n), rng)
        Xs.append(pts); ys.append(np.full(len(pts), i))
    X = np.vstack(Xs) + rng.normal(0, 0.15, (sum(counts), 2))
    y = np.concatenate(ys)
    print(f'--- nonconvex_{nm}  counts={counts}  areas={[round(p.area) for p in polys]}', flush=True)
    for m, p in [('IZGC-Full', dict(grid_size=2.0, R=30, min_size=5, buffer=0)),
                 ('IZGC-Full', dict(grid_size=3.0, R=30, min_size=5, buffer=0)),
                 ('K-means++', dict()),
                 ('DBSCAN', dict(eps=4.0, min_samples=5)),
                 ('GMM', dict(covariance_type='full', n_init=1))]:
        lab, _ = ad.run(m, X, 2, p, scene=dict(polys=polys, barrier_line=None), polygonise=False)
        k = len(set(int(v) for v in lab) - {-1})
        print(f'   {m:10s} {p}  ARI={adjusted_rand_score(y, lab):.3f}  k={k}', flush=True)
