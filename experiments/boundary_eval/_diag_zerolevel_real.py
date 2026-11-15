"""Same probe on the benchmark datasets, at grid steps inside the paper's range."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from algorithm_comparison.utils import load_dataset, normalize_coordinates


def build(X, g, R=30.0):
    tree = cKDTree(X)
    dens = np.array([len(tree.query_ball_point(p, R)) for p in X], float)
    min_x, max_x = X[:, 0].min(), X[:, 0].max()
    min_y, max_y = X[:, 1].min(), X[:, 1].max()
    i0, i1 = min_x - 2 * g, max_x + 2 * g
    j0, j1 = min_y - 2 * g, max_y + 2 * g

    cells = {(int((x - i0) // g), int((y - j0) // g)) for x, y in X}
    vp = []
    nx = int(np.ceil((i1 - i0) / g))
    ny = int(np.ceil((j1 - j0) / g))
    for a in range(nx):
        for b in range(ny):
            if (a, b) not in cells:
                vp.append((i0 + (a + 0.5) * g, j0 + (b + 0.5) * g))
    vp = np.array(vp) if vp else np.zeros((0, 2))

    P = np.vstack([X, vp])
    Z = np.concatenate([dens, np.zeros(len(vp))])
    xt = np.arange(i0, i1 + g, g)
    yt = np.arange(j0, j1 + g, g)
    GX, GY = np.meshgrid(xt, yt)
    zi = griddata((P[:, 0], P[:, 1]), Z, (GX, GY), method="linear")
    zi = np.nan_to_num(zi, nan=0.0)
    return GX, GY, zi, dens, len(vp)


def rings(cs, level):
    """closed rings at one level: count, total area, area-weighted mean area"""
    try:
        segs = cs.allsegs[list(cs.levels).index(level)]
    except ValueError:
        return 0, 0.0
    n, tot = 0, 0.0
    for s in segs:
        if len(s) >= 3 and np.allclose(s[0], s[-1]):
            x, y = s[:, 0], s[:, 1]
            tot += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            n += 1
    return n, tot


for name in ("jain", "spiral", "flame", "aggregation", "complex9", "2d_3c", "R15"):
    try:
        X, y = load_dataset(f"data/{name}.csv")
    except Exception as e:
        print(name, "load failed", e)
        continue
    X = normalize_coordinates(X)
    k = len(np.unique(y)) if y is not None else -1
    print(f"\n##### {name}  (n={len(X)}, K_true={k})")
    for g in (1.742, 2.826, 3.906):
        GX, GY, zi, dens, nvp = build(X, g)
        fig, ax = plt.subplots()
        cs = ax.contour(GX, GY, zi, levels=10)
        plt.close(fig)
        lv = np.asarray(cs.levels)
        n0, a0 = rings(cs, 0.0)
        has0 = bool(np.isclose(lv, 0.0).any())
        print(f"  g={g:5.3f} nodes={zi.size:5d} zeros={int((zi==0).sum()):5d} vp={nvp:5d} "
              f"zmin={zi.min():.2e} zmax={zi.max():7.2f} | levels={len(lv)} has0={has0} "
              f"lowest={lv.min():7.3f} | rings@0={n0} area@0={a0:8.1f}")
        if has0:
            print(f"        levels={np.round(lv, 2).tolist()}")
