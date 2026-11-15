"""Probe: which isoline level does the reference implementation actually extract?

Replicates engine steps of ContourBasedClusteringWithoutZ22.generate_contour_areas
(density -> virtual zero points at cell centres -> linear griddata onto grid nodes
-> ax.contour(levels=10)) and reports the levels matplotlib chooses.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from matplotlib.path import Path


def blobs(seed=0, per=120):
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.normal([25, 25], 5, size=(per, 2)),
        rng.normal([70, 30], 4, size=(per, 2)),
        rng.normal([50, 70], 7, size=(per, 2)),
    ])
    return X


def build(X, g, R=30.0):
    tree = cKDTree(X)
    dens = np.array([len(tree.query_ball_point(p, R)) for p in X], float)
    min_x, max_x = X[:, 0].min(), X[:, 0].max()
    min_y, max_y = X[:, 1].min(), X[:, 1].max()
    i0, i1 = min_x - 2 * g, max_x + 2 * g
    j0, j1 = min_y - 2 * g, max_y + 2 * g

    cells = set()
    for x, y in X:
        cells.add((int((x - i0) // g), int((y - j0) // g)))
    vp = []
    nx = int(np.ceil((i1 - i0) / g))
    ny = int(np.ceil((j1 - j0) / g))
    for a in range(nx):
        for b in range(ny):
            if (a, b) not in cells:
                vp.append((i0 + (a + 0.5) * g, j0 + (b + 0.5) * g, 0.0))
    vp = np.array(vp)

    P = np.vstack([X, vp[:, :2]])
    Z = np.concatenate([dens, np.zeros(len(vp))])

    xt = np.arange(i0, i1 + g, g)
    yt = np.arange(j0, j1 + g, g)
    GX, GY = np.meshgrid(xt, yt)
    zi = griddata((P[:, 0], P[:, 1]), Z, (GX, GY), method="linear")
    zi = np.nan_to_num(zi, nan=0.0)
    return GX, GY, zi, dens, P, Z


for g in (1.742, 2.826, 5.0):
    X = blobs()
    GX, GY, zi, dens, P, Z = build(X, g)
    fig, ax = plt.subplots()
    cs = ax.contour(GX, GY, zi, levels=10)
    plt.close(fig)
    lv = np.asarray(cs.levels)
    print(f"\n--- grid step g={g}")
    print(f"  density of actual points: min={dens.min():.0f} max={dens.max():.0f}")
    print(f"  field z: min={zi.min():.4f} max={zi.max():.4f}  (#nodes at 0: {int((zi==0).sum())}/{zi.size})")
    print(f"  contour levels ({len(lv)}): {np.round(lv,3).tolist()}")
    print(f"  lowest level / max = {lv.min():.3f}/{zi.max():.3f} = {lv.min()/zi.max():.3f}")

    # geometry of the contour set
    npath = len(cs.get_paths())
    closed, open_ = 0, 0
    for p in cs.get_paths():
        v = p.vertices
        if len(v) < 3:
            continue
        if np.allclose(v[0], v[-1]):
            closed += 1
        else:
            open_ += 1
    print(f"  paths={npath} closed={closed} open={open_}")
    for lev, segs in zip(cs.levels, cs.allsegs):
        if len(segs) == 0:
            continue
        areas = []
        for s in segs:
            if len(s) >= 3 and np.allclose(s[0], s[-1]):
                x, y = s[:, 0], s[:, 1]
                areas.append(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        print(f"    level {lev:7.3f}: {len(segs)} segment(s), closed={len(areas)}, "
              f"closed-area sum={sum(areas):9.1f}")

# what happens with an explicit zero level
X = blobs()
GX, GY, zi, dens, P, Z = build(X, 2.826)
fig, ax = plt.subplots()
for lev in ([0.0], [1.0]):
    cs = ax.contour(GX, GY, zi, levels=lev)
    print(f"\nexplicit levels={lev}: #segs={[len(s) for s in cs.allsegs]}")
plt.close(fig)
