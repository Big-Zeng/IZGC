"""Uniform adapters for every method compared in the boundary experiment.

Two things are deliberately identical across methods:

1. **Parameter search.** Every method is tuned with the grid published in the
   manuscript (supplementary Table S4) and the step sizes used by
   `algorithm_comparison/run_all.py`, on a validation seed, with ARI as the
   objective (the paper's `OPTIMIZE_METRIC`).  ARI carries no boundary
   information, so the boundary scores are never used to pick parameters.
   IZGC searches only the grid step, as in the paper, with R = 30 and
   theta = 5 fixed.

2. **Output form.** IZGC contributes its accepted contours; every point-label
   method is polygonised with the same alpha-shape routine whose alpha is
   chosen on the validation seed.  No method is compared in a form it would
   not itself produce.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import threading
import warnings

import numpy as np
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ContourBasedClusteringWithoutZ22 as CC  # noqa: E402
import regions as rg  # noqa: E402
from algorithm_comparison.alg_bang import run_bang  # noqa: E402
from algorithm_comparison.alg_clique2 import run_clique2  # noqa: E402
from algorithm_comparison.alg_dbscan import run_dbscan  # noqa: E402
from algorithm_comparison.alg_gmm import run_gmm  # noqa: E402
from algorithm_comparison.alg_kmeans import run_kmeans  # noqa: E402

IZGC_VARIANTS = ("IZGC-Full", "IZGC-NoZP")
POINT_METHODS = ("K-means++", "GMM", "DBSCAN", "CLIQUE", "BANG", "DPC-FWSN")
METHODS = IZGC_VARIANTS + POINT_METHODS
# methods that need a persistent MATLAB engine; they must run in one process,
# never inside a worker pool (each worker would boot its own MATLAB)
MATLAB_METHODS = ("DPC-FWSN",)

ALPHA_GRID = (0.05, 0.10, 0.20, 0.35)   # inverse length units

# --------------------------------------------------------------------------- #
# parameter ranges: supplementary Table S4, steps as in run_all.py
# --------------------------------------------------------------------------- #
IZGC_R = 30.0          # fixed in the paper
IZGC_THETA = 5         # fixed in the paper
IZGC_GRID = (0.5, 5.0, 0.001)         # Delta_g, main text Section 3.5

RANGES = {
    "DBSCAN": dict(eps=(1.0, 50.0, 1.0), min_samples=(2, 50, 1)),
    "CLIQUE": dict(intervals=(4, 100, 1), density_threshold=(1, 50, 1)),
    # Table S4 prints [4,50] for BANG levels, but the reference implementation
    # (pyclustering 0.10) needs 6 s per call at 20 levels, 12 s at 30 and more
    # than 25 s at 40 on an 800-point scene, so the search is capped at 20.
    # The comment inside algorithm_comparison/alg_bang.py states the same
    # usable range (3-15), so the cap follows the method's own practice.
    "BANG": dict(levels=(4, 20, 1), density_threshold=(0.01, 1.0, 0.05)),
    "GMM": dict(n_init=(1, 50, 5)),
    "K-means++": dict(n_init=(10, 10, 1), max_iter=(300, 300, 1)),
    # DPC-FWSN: k = number of neighbours (run_all.py uses 1..50 step 1); k = 1
    # leaves tempk1 empty inside the MATLAB wrapper, so the sweep starts at 2.
    "DPC-FWSN": dict(k=(2, 50, 1)),
}
GMM_COV = ("full", "tied", "diag", "spherical")


def _seq(start, end, step):
    # floor, not round: a range such as (1, 50, 5) must stop at 46, and
    # (0.01, 1.0, 0.05) must stop at 0.96, never overshoot the published bound
    n = int(math.floor((end - start) / step + 1e-9)) + 1
    return [start + i * step for i in range(n)]


def param_grid(name):
    """Full search grid of one method, in the order the paper defines it."""
    if name.startswith("IZGC"):
        g0, g1, gs = IZGC_GRID
        return [dict(grid_size=round(g, 3), R=IZGC_R, min_size=IZGC_THETA, buffer=0)
                for g in _seq(g0, g1, gs)]
    if name == "K-means++":
        return [dict()]                     # k is fixed to the true count
    if name == "GMM":
        r = RANGES["GMM"]
        return [dict(covariance_type=c, n_init=int(n))
                for c in GMM_COV for n in _seq(*r["n_init"])]
    if name == "DBSCAN":
        r = RANGES["DBSCAN"]
        return [dict(eps=round(e, 3), min_samples=int(m))
                for e in _seq(*r["eps"]) for m in _seq(*r["min_samples"])]
    if name == "CLIQUE":
        r = RANGES["CLIQUE"]
        return [dict(intervals=int(l), thr=int(t))
                for l in _seq(*r["intervals"]) for t in _seq(*r["density_threshold"])]
    if name == "BANG":
        r = RANGES["BANG"]
        return [dict(levels=int(l), thr=round(t, 3))
                for l in _seq(*r["levels"]) for t in _seq(*r["density_threshold"])]
    if name == "DPC-FWSN":
        return [dict(k=int(v)) for v in _seq(*RANGES["DPC-FWSN"]["k"])]
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# IZGC
# --------------------------------------------------------------------------- #
class _FastIZGC(CC.ContourBasedClusteringWithoutZ):
    """The reference algorithm with the point-in-contour test vectorised.

    The reference version calls `is_point_in_region` once per point per
    contour, which dominates the runtime.  Here the same interior test is
    applied to all points at once; the labelling rule is unchanged and the two
    implementations agree exactly (ARI = 1.0 on the test scenes).
    """

    def generate_contour_areas(self):
        from matplotlib.path import Path as MplPath
        from scipy.interpolate import griddata

        _, xt, yt = self.generate_virtual_points()
        allp = self.actual_points + self.virtual_points
        x1 = np.array([p[0] for p in allp])
        y1 = np.array([p[1] for p in allp])
        z1 = np.array([p[2] for p in allp])
        X, Y = np.meshgrid(xt, yt)
        zi = np.nan_to_num(griddata((x1, y1), z1, (X, Y), method="linear"), nan=0.0)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        cs = ax.contour(X, Y, zi, levels=10, colors="black")
        plt.close(fig)

        pts = np.array([[p[0], p[1]] for p in self.actual_points], float)
        areas = []
        for poly in self.line_to_polygon(cs):
            try:
                mask = MplPath(np.asarray(poly, float)).contains_points(pts)
            except Exception:
                continue
            if mask.any():
                areas.append(self.ContourArea(poly, [p for p, keep in zip(self.actual_points, mask) if keep]))

        self.contour_areas = areas
        # Nesting test: the reference routine compares every pair of contours
        # with shapely, which is O(k^2) full polygon predicates.  At a fine grid
        # step k reaches the thousands and that dominates the runtime, so the
        # pairs are rejected first by bounding box.  The parent relation itself
        # is unchanged: a contour can only lie inside another if its bounding
        # box does.
        geoms = [Polygon(a.polygon) for a in areas]
        bbox = [g.bounds for g in geoms]
        for i, a in enumerate(areas):
            ax0, ay0, ax1, ay1 = bbox[i]
            for j, b in enumerate(areas):
                if i == j:
                    continue
                bx0, by0, bx1, by1 = bbox[j]
                if ax0 < bx0 or ay0 < by0 or ax1 > bx1 or ay1 > by1:
                    continue
                if geoms[i].within(geoms[j]):
                    a.parent = b
                    b.isLeave = False
                    a.childLength += 1
        return areas


class _NoAutoZP(_FastIZGC):
    """Ablation: remove the automatic background zero-value points.

    The grid ticks are still returned so the interpolated field keeps the same
    support; only the fictitious z = 0 points disappear.
    """

    def generate_virtual_points(self):
        g = self.grid_size
        xs = [p[0] for p in self.actual_points]
        ys = [p[1] for p in self.actual_points]
        i0, i1 = min(xs) - 2 * g, max(xs) + 2 * g
        j0, j1 = min(ys) - 2 * g, max(ys) + 2 * g
        self.virtual_points = []
        self.interactive = False
        return [], np.arange(i0, i1 + g, g), np.arange(j0, j1 + g, g)


@contextlib.contextmanager
def _quiet_and_no_plot():
    """Silence prints and skip the 600-dpi figure dump inside IZGC."""
    buf = io.StringIO()
    orig_save = CC.plt.savefig
    orig_show = CC.plt.show
    CC.plt.savefig = lambda *a, **k: None
    CC.plt.show = lambda *a, **k: None
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        CC.plt.savefig = orig_save
        CC.plt.show = orig_show


def run_izgc(X, k, params, manual_points=None, auto_zp=True, seed=0, polygonise=True):
    from scipy.spatial import cKDTree

    grid_size = float(params["grid_size"])
    radius = float(params["R"])
    min_size = int(params["min_size"])
    buffer_distance = int(params.get("buffer", 0))

    tree = cKDTree(X)
    density = np.array([len(tree.query_ball_point(p, radius)) for p in X], float)
    combined = [(*X[i], density[i], i) for i in range(len(X))]

    cls_fn = _FastIZGC if auto_zp else _NoAutoZP
    kwargs = dict(grid_size=grid_size, max_buffer_distance=buffer_distance,
                  manual_virtual_points=manual_points or [],
                  min_cluster_size=min_size, interactive=False,
                  merge_small_clusters=True)
    with _quiet_and_no_plot():
        cls = cls_fn(combined, **kwargs)
        cls.perform_clustering()
        labels = np.asarray(cls.get_predicted_labels())
        raw = []
        if polygonise:
            for area in cls.density_centers:
                try:
                    p = Polygon(np.asarray(area.polygon, float)).buffer(0)
                    if not p.is_empty and p.area > 0:
                        raw.append(p)
                except Exception:
                    continue
    if raw:
        amax = max(p.area for p in raw)
        polys = [p for p in raw if p.area >= 0.02 * amax]
    else:
        polys = []
    return labels, polys


# --------------------------------------------------------------------------- #
# point-label methods
# --------------------------------------------------------------------------- #
def _as_labels(out, n):
    if out is None:
        raise RuntimeError("method unavailable (missing dependency)")
    lab = np.asarray(out)
    if lab.ndim > 1:
        lab = lab.ravel()
    if len(lab) != n:
        raise RuntimeError(f"label length {len(lab)} != {n}")
    return lab.astype(int)


def _polygonise(X, labels, alpha):
    polys = []
    for v in sorted(set(labels)):
        if v < 0:
            continue
        pts = X[labels == v]
        if len(pts) < 3:
            continue
        g = rg.alpha_shape(pts, alpha)
        if not g.is_empty and g.area > 0:
            polys.append(g)
    return polys


# --------------------------------------------------------------------------- #
# DPC-FWSN (MATLAB, R2022b engine)
# --------------------------------------------------------------------------- #
_DPC_ENGINE = None
_DPC_LOCK = threading.Lock()


def _dpc_engine():
    """One MATLAB session per process, started on first use."""
    global _DPC_ENGINE
    if _DPC_ENGINE is None:
        import matlab.engine
        # -singleCompThread: without it each MATLAB session grabs every core
        # for BLAS and starves the parallel IZGC sweep running beside it
        eng = matlab.engine.start_matlab("-singleCompThread")
        eng.addpath(os.path.join(ROOT, "compare", "algorithm"), nargout=0)
        _DPC_ENGINE = eng
    return _DPC_ENGINE


def run_dpc_fwsn(X, n_clusters, k_neighbors):
    """DPC-FWSN via the MATLAB wrapper `run_algorithm_pr_unsupervised`."""
    import matlab
    k = int(max(2, min(k_neighbors, len(X) - 1)))
    with _DPC_LOCK:
        eng = _dpc_engine()
        lab = eng.run_algorithm_pr_unsupervised(
            matlab.double(np.asarray(X, float).tolist()),
            float(n_clusters), float(k), nargout=1)
    return np.asarray(lab, float).ravel().astype(int)


def run_point_method(name, X, k, params, alpha=0.1, seed=0, polygonise=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if name == "K-means++":
            labels = run_kmeans(X, n_clusters=k, random_state=seed)
        elif name == "GMM":
            labels = run_gmm(X, n_components=k,
                             covariance_type=params.get("covariance_type", "full"),
                             n_init=int(params.get("n_init", 3)))
        elif name == "DBSCAN":
            labels = run_dbscan(X, eps=float(params["eps"]),
                                min_samples=int(params["min_samples"]))
        elif name == "CLIQUE":
            labels = run_clique2(X, intervals=int(params["intervals"]),
                                 density_threshold=int(params["thr"]))
        elif name == "BANG":
            labels = run_bang(X, levels=int(params["levels"]),
                              density_threshold=float(params["thr"]))
        elif name == "DPC-FWSN":
            labels = run_dpc_fwsn(X, n_clusters=k, k_neighbors=int(params["k"]))
        else:
            raise ValueError(name)
    labels = _as_labels(labels, len(X))
    return labels, (_polygonise(X, labels, alpha) if polygonise else [])


def run(name, X, k, params, scene=None, seed=0, polygonise=True):
    """Dispatch. Returns (labels, polygons)."""
    if name.startswith("IZGC"):
        manual = None
        if name == "IZGC-Full" and scene is not None:
            from scene_gen import barrier_zero_points
            manual = barrier_zero_points(scene)
        return run_izgc(X, k, params, manual_points=manual,
                        auto_zp=(name != "IZGC-NoZP"), seed=seed,
                        polygonise=polygonise)
    return run_point_method(name, X, k, params, alpha=params.get("alpha", 0.1),
                            seed=seed, polygonise=polygonise)
