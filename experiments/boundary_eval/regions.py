"""Region and boundary geometry shared by every method.

Everything works in the same scene coordinate system (0-100 square), on the
same grid and with the same boundary tolerance, which is what makes the
comparison between raster-based and point-based clusters meaningful.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


class Grid:
    def __init__(self, domain=(0.0, 100.0, 0.0, 100.0), step: float = 1.0):
        x0, x1, y0, y1 = domain
        self.step = float(step)
        self.xs = np.arange(x0 + step / 2, x1, step)
        self.ys = np.arange(y0 + step / 2, y1, step)
        self.nx, self.ny = len(self.xs), len(self.ys)
        self.shape = (self.ny, self.nx)
        self._gx, self._gy = np.meshgrid(self.xs, self.ys)
        self.area = step * step
        self.diag = float(np.hypot(x1 - x0, y1 - y0))

    def cells(self) -> np.ndarray:
        return np.c_[self._gx.ravel(), self._gy.ravel()]


def polygon_mask(geom, grid: Grid) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.zeros(grid.shape, bool)
    if geom.geom_type == "GeometryCollection":
        geom = unary_union([g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    m = shapely.contains_xy(geom, grid._gx, grid._gy)
    return np.asarray(m, bool)


def rasterize_labels(X: np.ndarray, labels: np.ndarray, grid: Grid,
                     fill_radius: float = 1.5) -> np.ndarray:
    """Majority label per cell; negative labels (noise) do not vote.

    A cell that contains only noise points stays background, so noise is
    removed from the region rather than being turned into area.
    """
    labels = np.asarray(labels)
    # IZGC numbers clusters from 0 while 0 also means "background" here:
    # remap every non-negative label to 1..K, keep -1 as noise
    uniq = sorted(u for u in np.unique(labels) if u >= 0)
    mapping = {u: i + 1 for i, u in enumerate(uniq)}
    labels = np.array([mapping.get(v, -1) if v >= 0 else -1 for v in labels], dtype=int)

    ix = np.floor((X[:, 0] - (grid.xs[0] - grid.step / 2)) / grid.step).astype(int)
    iy = np.floor((X[:, 1] - (grid.ys[0] - grid.step / 2)) / grid.step).astype(int)
    ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny) & (labels >= 0)
    ix, iy, lab = ix[ok], iy[ok], labels[ok]
    out = np.zeros(grid.shape, np.int32)
    if len(lab) == 0:
        return out
    key = iy * grid.nx + ix
    order = np.lexsort((lab, key))
    key, lab = key[order], lab[order]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    ends = np.r_[starts[1:], len(key)]
    for a, b in zip(starts, ends):
        vals, cnt = np.unique(lab[a:b], return_counts=True)
        out.flat[key[a]] = vals[np.argmax(cnt)]

    # Fill empty cells from the nearest labelled point, but only within one
    # cell: the point spacing (~1.5 units) leaves gaps that must be closed,
    # while a wider radius dilates the region and shifts the boundary outwards
    # (3 cells gave +27 % area and a systematically displaced outline).
    empty = out == 0
    if empty.any():
        pts = X[labels >= 0]
        if len(pts):
            tree = cKDTree(pts)
            cells = grid.cells()[empty.ravel()]
            dist, idx = tree.query(cells, k=1)
            vals = labels[labels >= 0][idx]
            vals = np.where(dist <= fill_radius * grid.step, vals, 0)
            flat = out.ravel()
            flat[np.flatnonzero(empty.ravel())] = vals
            out = flat.reshape(grid.shape)
    return out


def label_masks(labelmap: np.ndarray) -> dict[int, np.ndarray]:
    return {int(v): (labelmap == v) for v in np.unique(labelmap) if v > 0}


def mask_boundary_points(mask: np.ndarray, grid) -> np.ndarray:
    """Boundary cells of a mask, returned as scene coordinates."""
    cells = mask_boundary_cells(mask)
    if len(cells) == 0:
        return np.empty((0, 2))
    xs = grid.xs[0] + cells[:, 0] * grid.step
    ys = grid.ys[0] + cells[:, 1] * grid.step
    return np.c_[xs, ys]


def mask_boundary_cells(mask: np.ndarray) -> np.ndarray:
    """Cells of `mask` that touch a cell outside it (4-neighbourhood)."""
    if not mask.any():
        return np.empty((0, 2))
    # 4-neighbourhood: a 8-neighbour test makes the boundary band two cells
    # thick, which doubles the point count and depresses precision
    p = np.pad(mask, 1, constant_values=False)
    interior = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    edge = mask & ~interior
    ys, xs = np.nonzero(edge)
    return np.c_[xs, ys].astype(float)


# --------------------------------------------------------------------------- #
# alpha shapes (the common polygonisation applied to point-label methods)
# --------------------------------------------------------------------------- #
def alpha_shape(points: np.ndarray, alpha: float, min_frac: float = 0.05):
    """Alpha shape of a point set; `alpha` is in inverse length units.

    Small alpha -> tight, possibly fragmented hull; large alpha -> convex hull.
    Triangles whose circumradius exceeds 1/alpha are discarded.
    """
    points = np.asarray(points, float)
    if len(points) < 4:
        return Polygon(points).buffer(0) if len(points) >= 3 else Polygon()
    tri = Delaunay(points)
    keep = []
    for simplex in tri.simplices:
        p = points[simplex]
        area2 = abs(np.cross(p[1] - p[0], p[2] - p[0]))
        if area2 < 1e-9:
            continue
        a = np.linalg.norm(p[1] - p[2])
        b = np.linalg.norm(p[0] - p[2])
        c = np.linalg.norm(p[0] - p[1])
        radius = a * b * c / (2.0 * area2)
        if radius <= 1.0 / alpha:
            keep.append(Polygon(p))
    if not keep:
        return Polygon(points).convex_hull
    geom = unary_union(keep)
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    total = sum(g.area for g in parts)
    parts = [g for g in parts if g.area >= min_frac * total]
    return unary_union(parts) if len(parts) > 1 else parts[0]


def boundary_points(geom, step: float = 1.0) -> np.ndarray:
    """Densified points along every ring of a polygon (exterior and holes)."""
    if geom is None or geom.is_empty:
        return np.empty((0, 2))
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type == "MultiPolygon":
        geoms = list(geom.geoms)
    else:
        return np.empty((0, 2))
    out = []
    for g in geoms:
        for ring in [g.exterior, *g.interiors]:
            n = max(8, int(ring.length / step))
            out.append(np.array([(p.x, p.y) for p in
                                 (ring.interpolate(t) for t in np.linspace(0, ring.length, n))]))
    return np.vstack(out) if out else np.empty((0, 2))


def mask_to_geom(mask: np.ndarray, grid: Grid):
    """Cell mask -> union of cell squares (fallback when no polygon exists)."""
    if not mask.any():
        return Polygon()
    ys, xs = np.nonzero(mask)
    boxes = [shapely.box(x - grid.step / 2, y - grid.step / 2,
                         x + grid.step / 2, y + grid.step / 2)
             for x, y in zip(xs, ys)]
    return unary_union(boxes)


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def iou_matrix(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> tuple[list, list, np.ndarray]:
    gk, pk = sorted(gt), sorted(pred)
    M = np.zeros((len(gk), len(pk)))
    for i, g in enumerate(gk):
        for j, p in enumerate(pk):
            inter = np.logical_and(gt[g], pred[p]).sum()
            if inter:
                M[i, j] = inter / (gt[g].sum() + pred[p].sum() - inter)
    return gk, pk, M


def match_regions(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]):
    """Hungarian matching on IoU. Returns matched pairs and the full matrix."""
    gk, pk, M = iou_matrix(gt, pred)
    if M.size == 0:
        return [], gk, pk, M
    rows, cols = linear_sum_assignment(-M)
    pairs = [(gk[i], pk[j], float(M[i, j])) for i, j in zip(rows, cols) if M[i, j] > 0]
    return pairs, gk, pk, M


def nn_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.array([np.inf])
    return cKDTree(b).query(a, k=1)[0]
