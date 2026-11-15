"""Where does the V=0 contour run? Check whether its vertices sit on zero-valued nodes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from algorithm_comparison.utils import load_dataset, normalize_coordinates


def build(X, g, R=30.0):
    tree = cKDTree(X)
    dens = np.array([len(tree.query_ball_point(p, R)) for p in X], float)
    i0, i1 = X[:, 0].min() - 2 * g, X[:, 0].max() + 2 * g
    j0, j1 = X[:, 1].min() - 2 * g, X[:, 1].max() + 2 * g
    cells = {(int((x - i0) // g), int((y - j0) // g)) for x, y in X}
    vp = [(i0 + (a + 0.5) * g, j0 + (b + 0.5) * g)
          for a in range(int(np.ceil((i1 - i0) / g)))
          for b in range(int(np.ceil((j1 - j0) / g)))
          if (a, b) not in cells]
    vp = np.array(vp) if vp else np.zeros((0, 2))
    P = np.vstack([X, vp])
    Z = np.concatenate([dens, np.zeros(len(vp))])
    xt, yt = np.arange(i0, i1 + g, g), np.arange(j0, j1 + g, g)
    GX, GY = np.meshgrid(xt, yt)
    zi = np.nan_to_num(griddata((P[:, 0], P[:, 1]), Z, (GX, GY), method="linear"), nan=0.0)
    return GX, GY, zi, dens


X, y = load_dataset("data/spiral.csv")
X = normalize_coordinates(X)
GX, GY, zi, dens = build(X, 2.826)

fig, ax = plt.subplots()
cs = ax.contour(GX, GY, zi, levels=10)
plt.close(fig)

segs = cs.allsegs[list(cs.levels).index(0.0)]
print(f"zero-level segments: {len(segs)}")
for k, s in enumerate(segs[:3]):
    print(f"\nring {k}: {len(s)} vertices, closed={np.allclose(s[0], s[-1])}")
    dx = np.abs(s[:, 0][:, None] - GX[0][None, :])
    dy = np.abs(s[:, 1][:, None] - GY[:, 0][None, :])
    ni, nj = dx.argmin(1), dy.argmin(1)
    d = np.hypot(dx[np.arange(len(s)), ni], dy[np.arange(len(s)), nj])
    val = zi[nj, ni]
    print(f"  vertex->nearest node distance: max={d.max():.3e} mean={d.mean():.3e}")
    print(f"  value at those nodes: max|.|={np.abs(val).max():.3e}, all zero: {np.allclose(val, 0.0)}")

print("\nfield at nodes:")
print(f"  exact zeros: {int((zi == 0).sum())}, positive: {int((zi > 0).sum())}, "
      f"smallest positive: {zi[zi > 0].min():.4e}, largest: {zi.max():.2f}")
print(f"  number of distinct positive values: {len(np.unique(zi[zi > 0]))}")
