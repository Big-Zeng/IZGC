"""Publication figures for the six analytic boundary scenes.

Three sets of figures, each scene saved as an INDEPENDENT file so the panels
can be assembled in any order in the manuscript:

    01_ground_truth/  ground-truth labels + analytic boundary
    02_izgc_result/   IZGC-Full predicted labels (colours matched to truth)
    03_izgc_contour/  IZGC density-field contours + accepted cluster boundary

Additionally one 2x3 contact sheet per set is produced
(ground_truth_grid / izgc_result_grid / izgc_contour_grid).

Style follows street_test/results_overlap_izgc/plot_izgc_overlap_six_separate.py:
600 dpi, one square panel per file, large text.  Every single-panel figure has
the same physical size (no bbox_inches='tight'), so panels align exactly when
pasted together.

Usage:
    python plot_paper_figures.py                     # representative seed
    python plot_paper_figures.py --seed 0           # fixed seed
    python plot_paper_figures.py --dpi 600 --out paper_figures
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import adapters as ad          # noqa: E402
import scene_gen as sg         # noqa: E402


# --------------------------------------------------------------------------- #
# paper style (larger than the old fig_boundary.png, which used 8 pt)
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

CLUSTER_COLOURS = ["#3b6fb6", "#d1663a", "#4f9d69", "#8a63a8",
                   "#c9a227", "#5f6b7a"]
NOISE_COLOUR = "#8f8f8f"
GT_LINE = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.6, zorder=4)
CONTOUR_LINE = dict(colors="#9aa6b2", linewidths=0.7, alpha=0.75)
# 03 panel: keep only the outermost accepted contour of each cluster.
# Set the first switch to True to restore all 10 density levels, and the
# second to True to also show the small contours that IZGC later merged.
SHOW_DENSITY_CONTOURS = False
SHOW_MINOR_BOUNDARIES = False
# a single horizontal legend in the empty strip at the bottom of the frame
# (all six scenes have no data below y=14), so no point is ever covered
LEGEND_KW = dict(loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3,
                 framealpha=0.90, edgecolor="#cccccc", borderpad=0.35,
                 handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
                 labelspacing=0.3, fontsize=12.0)

SCENES = list(sg.SCENE_NAMES)
PARAMS_FILE = "_izgc0001_params.json"      # 0.001-step selected grid
SCORES_FILE = "results_izgc0001.csv"


# --------------------------------------------------------------------------- #
# IZGC: one call per scene, keeping the density field for the contour panel
# --------------------------------------------------------------------------- #
def run_izgc_keep_field(scene: dict, params: dict) -> dict:
    """Same computation as adapters.run_izgc, but retain the model object.

    The contour panel must show *the algorithm's own* contour set (10 levels
    of the interpolated density field with the automatic zero-value points),
    so the field is rebuilt from exactly the points the model used.
    """
    X = scene["X"]
    g = float(params["grid_size"])
    radius = float(params["R"])
    min_size = int(params["min_size"])
    buffer_distance = int(params.get("buffer", 0))

    tree = cKDTree(X)
    density = np.array([len(tree.query_ball_point(p, radius)) for p in X], float)
    combined = [(*X[i], density[i], i) for i in range(len(X))]

    with ad._quiet_and_no_plot():
        model = ad._FastIZGC(
            combined,
            grid_size=g,
            max_buffer_distance=buffer_distance,
            manual_virtual_points=[],
            min_cluster_size=min_size,
            interactive=False,
            merge_small_clusters=True,
        )
        model.perform_clustering()
        labels = np.asarray(model.get_predicted_labels(), dtype=int)

    # accepted boundaries: outermost contour areas, as in adapters.run_izgc
    raw = []
    for area in model.density_centers:
        poly = np.asarray(area.polygon, float)
        if len(poly) < 3:
            continue
        cx, cy = poly.mean(axis=0)
        raw.append((cx, poly))
    if raw:
        areas = [_poly_area(p) for _, p in raw]
        amax = max(areas)
        raw = [(cx, p) for (cx, p), a in zip(raw, areas)
               if amax <= 0 or a >= 0.02 * amax]
    # the two largest contours are the two clusters: order them left/right so
    # blue/orange always match the ground-truth panels.  Any remaining small
    # contour (later merged into a cluster) is drawn last, in gray.
    by_area = sorted(raw, key=lambda t: -_poly_area(t[1]))
    main = sorted(by_area[:2], key=lambda t: t[0])
    rest = sorted(by_area[2:], key=lambda t: t[0])
    boundaries = [p for _, p in main + rest]

    # rebuild the interpolation grid from the model's own points
    actual = np.array([[p[0], p[1], p[2]] for p in model.actual_points], float)
    virtual = np.array([[p[0], p[1], p[2]] for p in model.virtual_points], float)
    allp = np.vstack([actual, virtual])
    i0, i1 = actual[:, 0].min() - 2 * g, actual[:, 0].max() + 2 * g
    j0, j1 = actual[:, 1].min() - 2 * g, actual[:, 1].max() + 2 * g
    xt = np.arange(i0, i1 + g, g)
    yt = np.arange(j0, j1 + g, g)
    GX, GY = np.meshgrid(xt, yt)
    zi = np.nan_to_num(
        griddata((allp[:, 0], allp[:, 1]), allp[:, 2], (GX, GY), method="linear"),
        nan=0.0,
    )
    return dict(labels=labels, boundaries=boundaries, GX=GX, GY=GY, zi=zi,
                grid_size=g, n_virtual=len(virtual))


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def match_labels_to_truth(scene: dict, labels: np.ndarray) -> dict:
    """Colour predicted clusters by the ground-truth region they overlap most."""
    X, polys = scene["X"], scene["polys"]
    mapping = {}
    for lab in sorted({int(v) for v in labels}):
        if lab < 0:
            continue
        pts = X[labels == lab]
        cnt = [int(shapely.contains_xy(p, pts[:, 0], pts[:, 1]).sum()) for p in polys]
        j = int(np.argmax(cnt))
        mapping[lab] = j if cnt[j] > 0 else -1
    return mapping


def representative_seeds(scores: pd.DataFrame) -> dict:
    """Pick one *typical successful* run per scene.

    Rule: among the runs that recover the true cluster count, take the one
    whose ARI is closest to the scene's all-seed median ARI.  Falling back to
    all runs when no run has the correct K keeps the figure from showing a
    degenerate k=1 example.  The chosen seed and its scores are written into
    figure_manifest.json and printed in the caption, so nothing is hidden;
    --seed N reproduces any other single run.
    """
    out = {}
    for sc in SCENES:
        d = scores[(scores.scene == sc) & (scores.method == "IZGC-Full")]
        if not len(d):
            out[sc] = 0
            continue
        med = float(d.ARI.median())
        cand = d[d.k_pred == d.k_gt]
        if not len(cand) or float(cand.ARI.max()) < 0.6:
            cand = d
        cand = cand.assign(dist=(cand.ARI - med).abs())
        cand = cand.sort_values(["dist", "seed"])
        out[sc] = int(cand.iloc[0].seed)
    return out


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
# Panels are pasted together in the manuscript, so no per-panel heading above
# the frame (scene name / "Ground truth" / method name).  Set this to True to
# restore those headings on every panel.
SHOW_SCENE_TITLE = False
CAPTION_SIZE = 14.5


def style_axes(ax, caption: str, caption_size: float = CAPTION_SIZE,
               title: str | None = None):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", length=4, width=1.0, direction="out")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if SHOW_SCENE_TITLE and title:
        ax.set_title(title, fontsize=18, pad=10)
    # one short caption inside the top of the frame; no seed, no scene name,
    # no white box (the strip above y=90 is empty in all six scenes)
    ax.text(0.5, 0.975, caption, transform=ax.transAxes,
            ha="center", va="top", fontsize=caption_size, color="#222222",
            zorder=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def draw_gt_boundary(ax, scene):
    for p in scene["polys"]:
        geoms = [p] if p.geom_type == "Polygon" else list(p.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, **GT_LINE)


def gt_handles(k):
    h = [Line2D([0], [0], marker="o", linestyle="none", color=CLUSTER_COLOURS[i],
                markersize=7, label=f"Cluster {i + 1}") for i in range(k)]
    h.append(Line2D([0], [0], color="black", linestyle=(0, (5, 4)),
                    linewidth=1.6, label="GT boundary"))
    return h


def draw_ground_truth(ax, scene, caption_size=CAPTION_SIZE):
    X, y = scene["X"], scene["y"]
    for i in range(1, scene["k"] + 1):
        m = y == i
        ax.scatter(X[m, 0], X[m, 1], s=11, alpha=0.80,
                   c=CLUSTER_COLOURS[(i - 1) % len(CLUSTER_COLOURS)],
                   edgecolors="none", zorder=2)
    draw_gt_boundary(ax, scene)
    style_axes(ax, f"$K$ = {scene['k']},  $n$ = {len(X)}",
               caption_size=caption_size,
               title=f"{scene['name']} — Ground truth")
    ax.legend(handles=gt_handles(scene["k"]), **LEGEND_KW)


def draw_izgc_result(ax, scene, iz, ari, caption_size=CAPTION_SIZE):
    X, labels = scene["X"], iz["labels"]
    mapping = match_labels_to_truth(scene, labels)
    for lab in sorted({int(v) for v in labels}):
        m = labels == lab
        if lab < 0 or mapping.get(lab, -1) < 0:
            ax.scatter(X[m, 0], X[m, 1], s=13, c=NOISE_COLOUR, marker="x",
                       linewidths=0.9, alpha=0.75, zorder=2)
        else:
            ax.scatter(X[m, 0], X[m, 1], s=11, alpha=0.80,
                       c=CLUSTER_COLOURS[mapping[lab] % len(CLUSTER_COLOURS)],
                       edgecolors="none", zorder=2)
    draw_gt_boundary(ax, scene)
    k_pred = len({int(v) for v in labels if v >= 0})
    style_axes(ax,
               rf"$\Delta g$={iz['grid_size']:.3f}, "
               rf"$K_{{pred}}$={k_pred}, ARI={ari:.3f}",
               caption_size=caption_size,
               title=f"{scene['name']} — IZGC-Full")
    ax.legend(handles=gt_handles(scene["k"]), **LEGEND_KW)


def draw_izgc_contour(ax, scene, iz, caption_size=CAPTION_SIZE):
    X = scene["X"]
    # by default only the outermost accepted contour is drawn; the ten
    # nested density levels are available via SHOW_DENSITY_CONTOURS
    if SHOW_DENSITY_CONTOURS:
        ax.contour(iz["GX"], iz["GY"], iz["zi"], levels=10, **CONTOUR_LINE)
    ax.scatter(X[:, 0], X[:, 1], s=6, c="#4a4a4a", alpha=0.30,
               edgecolors="none", zorder=2)
    polys = iz["boundaries"] if SHOW_MINOR_BOUNDARIES else iz["boundaries"][:2]
    for i, poly in enumerate(polys):
        # first two = blue/orange outer boundaries, as in the GT panels;
        # the rest are small contours merged by IZGC, kept visible but muted
        main = i < 2
        c = CLUSTER_COLOURS[i] if main else "#8a97a5"
        ax.fill(poly[:, 0], poly[:, 1], color=c,
                alpha=0.07 if main else 0.03, zorder=1)
        ax.plot(poly[:, 0], poly[:, 1], color=c,
                linewidth=2.6 if main else 1.4,
                solid_capstyle="round", zorder=5)
    # reference geometry, identical to the ground-truth and clustering panels
    draw_gt_boundary(ax, scene)
    k_pred = len({int(v) for v in iz["labels"] if v >= 0})
    caption = rf"$\Delta g$={iz['grid_size']:.3f}, "
    if SHOW_DENSITY_CONTOURS:
        caption += "10 levels, "
    caption += rf"$K_{{pred}}$={k_pred}"
    style_axes(ax, caption, caption_size=caption_size,
               title=f"{scene['name']} — IZGC contours")
    handles = []
    if SHOW_DENSITY_CONTOURS:
        handles.append(Line2D([0], [0], color="#9aa6b2", linewidth=1.4,
                              label="Contours"))
    handles += [
        Line2D([0], [0], color=CLUSTER_COLOURS[0], linewidth=2.6,
               label="Outer boundary"),
        Line2D([0], [0], color=GT_LINE["color"],
               linestyle=GT_LINE["linestyle"],
               linewidth=GT_LINE["linewidth"], label="GT boundary"),
        Line2D([0], [0], marker="o", linestyle="none", color="#4a4a4a",
               markersize=4, alpha=0.6, label="Points"),
    ]
    # three longer labels: drop one point of size so the row stays inside the frame
    ax.legend(handles=handles, **{**LEGEND_KW, "fontsize": 11.0})


def save(fig, base: Path, dpi: int):
    # no bbox_inches='tight': every panel keeps exactly the same canvas size
    fig.savefig(base.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")


def one_panel(draw_fn, base: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    draw_fn(ax, caption_size=CAPTION_SIZE)
    fig.tight_layout(pad=0.8)
    save(fig, base, dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=-1,
                    help="-1 = representative (median-ARI) seed per scene")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip the three 2x3 contact sheets")
    args = ap.parse_args()

    out = Path(os.path.join(HERE, args.out))
    dirs = [out / "01_ground_truth", out / "02_izgc_result",
            out / "03_izgc_contour"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    params = json.load(open(os.path.join(HERE, PARAMS_FILE), encoding="utf-8"))
    scores = pd.read_csv(os.path.join(HERE, SCORES_FILE))
    izscores = scores[scores.method == "IZGC-Full"]
    seeds = representative_seeds(scores) if args.seed < 0 else \
        {sc: args.seed for sc in SCENES}

    # contact sheets: 2 rows x 3 columns, same order as SCENE_NAMES
    grids = []
    if not args.no_grid:
        for _ in range(3):
            fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.6))
            grids.append((fig, axes.ravel()))

    manifest = []
    for sc in SCENES:
        seed = seeds[sc]
        p = params[f"{sc}|IZGC-Full"]["params"]
        row = izscores[(izscores.scene == sc) & (izscores.seed == seed)]
        ari = float(row.ARI.iloc[0]) if len(row) else float("nan")

        scene = sg.build(sc, seed)
        iz = run_izgc_keep_field(scene, p)

        one_panel(lambda ax, caption_size, s=scene:
                  draw_ground_truth(ax, s, caption_size),
                  dirs[0] / f"gt_{sc}", args.dpi)
        one_panel(lambda ax, caption_size, s=scene, z=iz, a=ari:
                  draw_izgc_result(ax, s, z, a, caption_size),
                  dirs[1] / f"izgc_{sc}", args.dpi)
        one_panel(lambda ax, caption_size, s=scene, z=iz:
                  draw_izgc_contour(ax, s, z, caption_size),
                  dirs[2] / f"contour_{sc}", args.dpi)

        if grids:
            i = SCENES.index(sc)
            draw_ground_truth(grids[0][1][i], scene, caption_size=CAPTION_SIZE)
            draw_izgc_result(grids[1][1][i], scene, iz, ari, caption_size=CAPTION_SIZE)
            draw_izgc_contour(grids[2][1][i], scene, iz, caption_size=CAPTION_SIZE)

        manifest.append(dict(scene=sc, seed=seed, n=int(len(scene["X"])),
                             k_gt=int(scene["k"]), delta_g=iz["grid_size"],
                             k_pred=int(len({int(v) for v in iz["labels"] if v >= 0})),
                             ari=round(ari, 4), bf_alpha=round(float(row.BF_alpha.iloc[0]), 4)
                             if len(row) else None,
                             n_virtual_points=iz["n_virtual"],
                             n_boundary_polygons=len(iz["boundaries"])))
        print(f"{sc:13s} seed={seed:2d}  dg={iz['grid_size']:.3f}  "
              f"K_pred={manifest[-1]['k_pred']}  ARI={ari:.3f}  "
              f"contours={len(iz['boundaries'])}", flush=True)

    if grids:
        for (fig, _), name in zip(grids,
                                  ["grid_ground_truth", "grid_izgc_result",
                                   "grid_izgc_contour"]):
            fig.tight_layout(pad=1.4, h_pad=2.0, w_pad=1.0)
            save(fig, out / name, args.dpi)
            plt.close(fig)

    with open(out / "figure_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(dict(dpi=args.dpi, figsize_in=[5.0, 5.0],
                       params_file=PARAMS_FILE, scores_file=SCORES_FILE,
                       panels=manifest), fh, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
