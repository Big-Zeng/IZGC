"""Membership audit through the reference adapter (shapely available)."""
import numpy as np
from shapely.geometry import Point

from adapters import run_izgc
from algorithm_comparison.utils import load_dataset, normalize_coordinates

G = 2.826

for name in ("jain", "flame", "aggregation", "complex9", "spiral", "2d_3c", "R15"):
    try:
        X, y = load_dataset(f"../../data/{name}.csv")
    except Exception as exc:  # noqa: BLE001
        print(name, "skip", exc)
        continue
    X = normalize_coordinates(X)
    labels, polys = run_izgc(X, len(np.unique(y)) if y is not None else 0,
                             dict(grid_size=G, R=30.0, min_size=5, buffer=0))
    n = len(X)
    inside = np.zeros(n, int)
    onb = np.zeros(n, int)
    for p in polys:
        bd = p.boundary
        for i, (x0, y0) in enumerate(X):
            pt = Point(x0, y0)
            if p.contains(pt):
                inside[i] += 1
            elif bd.distance(pt) < 1e-12:
                onb[i] += 1
    lab = np.asarray(labels)
    print(f"{name:12s} n={n:5d} contours={len(polys):2d} "
          f"| in>1: {int((inside > 1).sum()):3d} "
          f"| on boundary: {int((onb > 0).sum()):3d} "
          f"| outside all: {int((inside == 0).sum()):4d} "
          f"| clusters: {len(np.unique(lab[lab >= 0])):2d} | noise: {int((lab == -1).sum()):3d}")
