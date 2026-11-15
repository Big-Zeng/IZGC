"""Analytic boundary scenes for the boundary-quality experiment.

Every ground-truth boundary is the polygon that was used to *generate* the
points, never a shape reconstructed from the predicted labels, so the
evaluation cannot be circular.

Six scenes in three families:

    A  non-convex   c : a C-shaped annulus whose bite is filled by cluster 2
                    u : a U-shaped block whose notch is filled by cluster 2
    B  narrow gap   gap_4 : two irregular blobs, 4.0 units apart
                    gap_2 : same, 2.0 units apart
    C  density      density_3 : heterogeneous density, contrast ~3x
                    density_8 : heterogeneous density, contrast ~8x

Each cluster holds 300-500 points and all coordinates live in a fixed 0-100
square, i.e. the same normalised frame the manuscript uses for every baseline,
so the paper's parameter ranges apply without rescaling.
"""

from __future__ import annotations

import json

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

DOMAIN = (0.0, 100.0, 0.0, 100.0)
N_MIN, N_MAX = 300, 500


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def disk(cx: float, cy: float, r: float, n: int = 96) -> Polygon:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])


def blob(cx: float, cy: float, r: float, rng, nv: int = 9, jitter: float = 0.30) -> Polygon:
    """A star-shaped irregular blob; fixed seed -> fixed shape."""
    ang = np.sort(rng.uniform(0, 2 * np.pi, nv))
    rad = r * (1.0 - jitter + 2.0 * jitter * rng.uniform(size=nv))
    poly = Polygon(np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def sector(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> Polygon:
    """Circular sector in degrees, used to cut and to refill a concavity."""
    a = np.deg2rad(np.linspace(a0, a1, n))
    return Polygon(np.r_[[[cx, cy]], np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]])


def sample_in(poly: Polygon, n: int, rng, density=None, batch: int = 8192) -> np.ndarray:
    """Rejection sampling inside `poly`; `density(x, y) -> acceptance in [0, 1]`."""
    minx, miny, maxx, maxy = poly.bounds
    out: list[np.ndarray] = []
    got, guard = 0, 0
    while got < n and guard < 4000:
        guard += 1
        cand = rng.uniform([minx, miny], [maxx, maxy], size=(batch, 2))
        cand = cand[shapely.contains_xy(poly, cand[:, 0], cand[:, 1])]
        if density is not None and len(cand):
            cand = cand[rng.uniform(size=len(cand)) < density(cand[:, 0], cand[:, 1])]
        if len(cand):
            take = cand[: n - got]
            out.append(take)
            got += len(take)
    if got < n:
        raise RuntimeError(f"only {got}/{n} points sampled; geometry too small")
    return np.vstack(out)


def uniform_scale(X: np.ndarray, span: float = 100.0) -> np.ndarray:
    """Scale x and y by ONE common factor so the aspect ratio is preserved.

    Replaces per-axis min-max normalisation, which stretches distances.
    """
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0)
    ext = X.max(axis=0) - lo
    s = span / max(ext.max(), 1e-12)
    return (X - lo) * s


# --------------------------------------------------------------------------- #
# scene families
# --------------------------------------------------------------------------- #
def scene_nonconvex(kind: str, rng) -> dict:
    """Two interlocking regions: a straight (Voronoi) cut cannot separate them.

    Cluster 2 fills the concavity of cluster 1, so any convex or linearly
    separable region model pays for the area it swallows.
    """
    # The inset of 4.5 units is not cosmetic: below ~4 units (about twice the
    # inter-point spacing) the two clouds touch, no density dip exists between
    # them, and even IZGC merges them, so the scene would measure nothing but
    # sampling noise.  At 4.5 the interlocking structure is resolvable while a
    # straight (Voronoi) cut still has to cross one of the two regions.
    if kind == "c":
        disc = disk(46, 50, 34)
        bite = box(46, 22, 92, 78)                   # rectangular bite from the rim
        body = disc.difference(bite)
        fill = bite.intersection(disc.buffer(1.0)).buffer(-4.5)
    elif kind == "u":
        # notch cut through the top edge, so the body is a true U with two
        # arms of comparable width and the second cluster sits in the trough
        hole = box(28, 38, 72, 92)
        body = box(26, 16, 86, 80).difference(hole)
        fill = box(28, 38, 72, 80).buffer(-4.5)
    else:
        raise ValueError(kind)
    return _pack(f"nonconvex_{kind}", "A", [body, fill], rng)


def scene_gap(gap: float, rng) -> dict:
    """Two irregular blobs separated by a corridor of exactly `gap`.

    The shapes are generated first and then translated so that their closest
    distance equals `gap`, which keeps the separation analytic instead of
    nominal (the irregular outline would otherwise make the true gap differ
    from the centre-to-centre arithmetic).
    """
    r = 16.0
    left = blob(0, 0, r, rng)
    right = blob(0, 0, r, rng)
    left = _shift(left, -left.centroid.x, 0)
    right = _shift(right, -right.centroid.x, 0)
    # push the right blob until the closest-point distance equals `gap`
    # (the nearest pair is generally not horizontal, so a single shift by the
    # bounding-box arithmetic overshoots); the distance is monotone in dx
    dx = left.bounds[2] + gap - right.bounds[0]
    base, placed = right, None
    for _ in range(6):
        moved = _shift(base, dx, 0)
        err = left.distance(moved) - gap
        if abs(err) < 1e-3:
            placed = moved
            break
        dx -= err * 1.02
    if placed is None:
        # a few seeds start from blobs whose nearest pair is not on the axis,
        # so the secant correction can overshoot into an overlap; fall back to
        # a bisection on dx, which cannot cross the two shapes
        placed = _separate_to_gap(base, left, gap)
    right = placed
    union = unary_union([left, right])
    minx, miny, maxx, maxy = union.bounds
    # centre the pair inside the 0-100 frame
    tx = (100.0 - (maxx - minx)) / 2.0 - minx
    ty = (100.0 - (maxy - miny)) / 2.0 - miny
    left, right = _shift(left, tx, ty), _shift(right, tx, ty)
    assert left.distance(right) > 0
    return _pack(f"gap_{gap:g}", "B", [left, right], rng,
                 gap=round(float(left.distance(right)), 2))


def _shift(poly: Polygon, dx: float, dy: float) -> Polygon:
    from shapely.affinity import translate
    return translate(poly, xoff=dx, yoff=dy)


def _separate_to_gap(base: Polygon, left: Polygon, gap: float) -> Polygon:
    """Translate `base` along +x until its distance to `left` is `gap`.

    Bisection, so the result is always separated (distance > 0) even when the
    shapes' nearest pair is not the bounding-box pair.  Only used when the
    secant loop in `scene_gap` fails to converge.
    """
    hi = 1.0
    for _ in range(80):
        if left.distance(_shift(base, hi, 0)) >= gap:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if left.distance(_shift(base, mid, 0)) >= gap:
            hi = mid
        else:
            lo = mid
        if hi - lo < 5e-4:
            break
    return _shift(base, hi, 0)


def scene_density(ratio: float, rng) -> dict:
    """Same two-cluster layout at different mean densities, plus an internal
    gradient in the sparse cluster so density is heterogeneous *within* a
    cluster as well as between clusters.

    `ratio` is the target between-cluster density contrast (points per unit
    area); the achieved value is measured and stored in the manifest.
    """
    if ratio <= 4:                       # ~3x, both clusters mid-sized
        r_lo, n_lo, r_hi, n_hi = 24.0, 340, 14.0, 340
        cx_lo, cx_hi = 30.0, 78.0
    else:                                # ~8x, large sparse vs small dense
        r_lo, n_lo, r_hi, n_hi = 27.0, 320, 10.5, 400
        cx_lo, cx_hi = 30.0, 80.0
    sparse = disk(cx_lo, 50, r_lo)
    dense = disk(cx_hi, 50, r_hi)

    # within the sparse cluster the rim is `inner` times denser than the core
    inner = 2.0

    def grad(x, y):
        t = np.clip(np.hypot(x - cx_lo, y - 50.0) / r_lo, 0.0, 1.0)
        return np.clip(inner ** (t - 1.0), 0.05, 1.0)

    X_a = sample_in(sparse, n_lo, rng, density=grad)
    X_b = sample_in(dense, n_hi, rng)
    X = np.vstack([X_a, X_b])
    y = np.r_[np.ones(len(X_a), int), np.full(len(X_b), 2)]

    a_area = sparse.area
    b_area = dense.area
    measured = (len(X_b) / b_area) / (len(X_a) / a_area)
    return _finish(f"density_{ratio:g}", "C", X, y, [sparse, dense], rng,
                   target_ratio=ratio, measured_ratio=float(measured))


def _pack(name: str, family: str, polys: list[Polygon], rng, **meta) -> dict:
    """Sample every region at ONE common points-per-area rate, 300-500 points.

    The smallest region gets 300 points (the lower bound requested) and every
    other region receives points in proportion to its area, clipped at 500.
    This keeps the two clusters equally dense, so a non-convex test measures
    geometry rather than a density contrast.
    """
    areas = np.array([p.area for p in polys])
    counts = np.clip(np.round(N_MIN * areas / areas.min()), N_MIN, N_MAX).astype(int)
    Xs, ys = [], []
    for i, (p, n) in enumerate(zip(polys, counts), 1):
        pts = sample_in(p, int(n), rng)
        Xs.append(pts)
        ys.append(np.full(len(pts), i))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return _finish(name, family, X, y, polys, rng, **meta)


def _finish(name, family, X, y, polys, rng, **meta) -> dict:
    # a small jitter breaks the sampler's exact lattice; it is an order of
    # magnitude below the boundary tolerance
    X = X + rng.normal(0.0, 0.15, size=X.shape)
    polys = [Polygon(p.exterior.coords) if p.geom_type == "Polygon" else p for p in polys]
    return dict(
        name=name,
        family=family,
        X=X,
        y=y,
        polys=polys,
        k=len(polys),
        counts=[int((y == i).sum()) for i in range(1, len(polys) + 1)],
        barrier_line=None,
        barrier_band=None,
        **meta,
    )


SCENES = [
    ("nonconvex_c", lambda r: scene_nonconvex("c", r)),
    ("nonconvex_u", lambda r: scene_nonconvex("u", r)),
    ("gap_4", lambda r: scene_gap(4.0, r)),
    ("gap_2", lambda r: scene_gap(2.0, r)),
    ("density_3", lambda r: scene_density(3.0, r)),
    ("density_8", lambda r: scene_density(8.0, r)),
]
SCENE_NAMES = [s[0] for s in SCENES]
FAMILY_OF = {nm: {"nonconvex": "A", "gap": "B", "density": "C"}[nm.split("_")[0]]
             for nm in SCENE_NAMES}


def build(name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    for key, fn in SCENES:
        if key == name:
            s = fn(rng)
            s["seed"] = seed
            return s
    raise KeyError(f"unknown scene {name}")


def barrier_zero_points(scene: dict, step: float = 2.0) -> list[tuple[float, float]]:
    """Manual zero-value points along the barrier centre line (family D only).

    The six scenes used in this round contain no barrier, so this returns an
    empty list; it is kept so the ablation hook stays available.
    """
    line = scene.get("barrier_line")
    if line is None:
        return []
    n = max(2, int(line.length / step))
    return [(p.x, p.y) for p in (line.interpolate(t) for t in np.linspace(0, line.length, n))]


def manifest(path: str = "scene_manifest.json") -> list[dict]:
    rows = []
    for nm in SCENE_NAMES:
        s = build(nm, 0)
        row = dict(scene=nm, family=s["family"], k=s["k"], n=len(s["X"]),
                   counts=s["counts"],
                   area=[round(p.area, 1) for p in s["polys"]],
                   density=[round(c / p.area, 3) for c, p in zip(s["counts"], s["polys"])],
                   gap=s.get("gap"), target_ratio=s.get("target_ratio"),
                   measured_ratio=None if s.get("measured_ratio") is None
                   else round(s["measured_ratio"], 2),
                   perimeter=[round(p.exterior.length, 1) for p in s["polys"]])
        rows.append(row)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)
    return rows


if __name__ == "__main__":
    for r in manifest():
        print(f"{r['scene']:14s} {r['family']}  k={r['k']}  n={r['n']:4d}  "
              f"per-cluster={r['counts']}  area={r['area']}  density={r['density']}"
              + (f"  gap={r['gap']}" if r["gap"] else "")
              + (f"  ratio={r['measured_ratio']}" if r["measured_ratio"] else ""))
