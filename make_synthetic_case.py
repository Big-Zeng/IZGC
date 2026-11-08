"""Generate the synthetic stand-ins used for the case-study demonstration.

The municipal delivery-point sets analysed in the paper were supplied by a
commercial partner and cannot be redistributed, so this script produces point
sets with comparable size and structure that can be published alongside the
code.  Nothing here is derived from the original coordinates: only aggregate
descriptors of each case area (point count, number of density clusters, spatial
extent, typical cluster compactness and background scatter) were used to set
the parameters, and all geometry is drawn at random.

Coordinates are given in metres on an arbitrary local frame (origin at 0, 0).
They are not projected map coordinates of any real location.

Usage
-----
    python make_synthetic_case.py            # writes data/case_study_synthetic/
"""

import os

import numpy as np

OUT_DIR = os.path.join("data", "case_study_synthetic")

# name, n points, n density clusters, spatial extent (m), cluster radius as a
# fraction of the extent, median cluster size, background-scatter fraction
CASE_SETS = [
    # dense agglomerations around a large urban core
    dict(name="synthetic_dense_urban", n=2545, k=57, extent=(41_000, 48_000),
         spread_frac=0.0055, size_median=12, noise=0.16, seed=101),
    # compact clusters of intermediate size
    dict(name="synthetic_compact_clustered", n=1093, k=25, extent=(56_000, 50_000),
         spread_frac=0.0050, size_median=10, noise=0.12, seed=102),
    # several large agglomerations plus small satellites
    dict(name="synthetic_polycentric", n=1232, k=25, extent=(50_000, 47_000),
         spread_frac=0.0050, size_median=9, noise=0.07, seed=103),
    # few, well separated and regular agglomerations
    dict(name="synthetic_regular_agglomerations", n=484, k=5, extent=(58_000, 60_000),
         spread_frac=0.0105, size_median=60, noise=0.12, seed=104),
    # heterogeneous densities with a high share of isolated points
    dict(name="synthetic_heterogeneous_noisy", n=487, k=32, extent=(21_000, 22_000),
         spread_frac=0.0070, size_median=8, noise=0.35, seed=105),
    # sparse, dispersed settlements
    dict(name="synthetic_sparse_dispersed", n=472, k=13, extent=(42_000, 41_000),
         spread_frac=0.0060, size_median=8, noise=0.18, seed=106),
]

# points tracing a barrier such as a river or an escarpment, used by the
# constraint-injection example
ZERO_POINTS = dict(name="synthetic_barrier_zero_points", n=557,
                   length=14_000, width=180, seed=107)


def _cluster_centres(rng, k, extent, min_sep):
    """Rejection-sample k centres that stay at least min_sep apart."""
    centres = []
    for _ in range(k):
        for _attempt in range(500):
            c = rng.uniform([0, 0], extent)
            if all(np.hypot(*(c - o)) > min_sep for o in centres):
                break
        centres.append(c)
    return np.array(centres)


def make_case(n, k, extent, spread_frac, size_median, noise, seed):
    rng = np.random.default_rng(seed)
    extent = np.asarray(extent, dtype=float)
    spread = spread_frac * extent.max()

    n_bg = int(round(n * noise))
    n_cl = n - n_bg

    centres = _cluster_centres(rng, k, extent, min_sep=spread * 2.0)

    # heavy-tailed cluster sizes: many small groups, a few large ones
    w = rng.lognormal(mean=np.log(size_median), sigma=0.9, size=k)
    sizes = np.maximum(2, np.round(w / w.sum() * n_cl).astype(int))
    # hand out or take back the rounding remainder
    order = np.argsort(-w)
    i = 0
    while sizes.sum() < n_cl:
        sizes[order[i % k]] += 1
        i += 1
    while sizes.sum() > n_cl:
        j = order[-1 - (i % k)]
        if sizes[j] > 2:
            sizes[j] -= 1
        i += 1

    pts = []
    for c, m in zip(centres, sizes):
        # random anisotropy and orientation per cluster
        theta = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        s = spread * rng.uniform(0.5, 1.8, size=2) * np.array([1.0, rng.uniform(0.3, 1.0)])
        blob = rng.normal(size=(m, 2)) * s
        pts.append(blob @ rot.T + c)

    background = rng.uniform([0, 0], extent, size=(n_bg, 2))
    X = np.vstack(pts + [background])
    rng.shuffle(X)
    return X


def make_barrier(n, length, width, seed):
    """A meandering ridge of points, standing in for a river or an escarpment."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, size=n))
    x = t * length
    y = sum(a * np.sin(2 * np.pi * f * t + p)
            for a, f, p in [(0.09 * length, 1.0, 0.4),
                            (0.05 * length, 2.0, 1.9),
                            (0.02 * length, 5.0, 3.1)])
    jitter = rng.normal(scale=width, size=(n, 2))
    return np.column_stack([x, y]) + jitter


def write_csv(path, X):
    np.savetxt(path, X, delimiter=",", header="x,y", comments="", fmt="%.3f")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for cfg in CASE_SETS:
        X = make_case(cfg["n"], cfg["k"], cfg["extent"], cfg["spread_frac"],
                      cfg["size_median"], cfg["noise"], cfg["seed"])
        assert len(X) == cfg["n"], (cfg["name"], len(X))
        write_csv(os.path.join(OUT_DIR, cfg["name"] + ".csv"), X)
        print("%-34s n=%5d  k=%3d" % (cfg["name"], len(X), cfg["k"]))

    Z = make_barrier(ZERO_POINTS["n"], ZERO_POINTS["length"],
                     ZERO_POINTS["width"], ZERO_POINTS["seed"])
    write_csv(os.path.join(OUT_DIR, ZERO_POINTS["name"] + ".csv"), Z)
    print("%-34s n=%5d" % (ZERO_POINTS["name"], len(Z)))


if __name__ == "__main__":
    main()
