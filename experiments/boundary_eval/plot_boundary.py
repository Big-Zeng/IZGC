"""Qualitative comparison figure: 6 scenes x (ground truth + 4 methods).

Every panel is drawn in the same 0-100 frame.  Points are coloured by the
method's predicted label; the dashed black line is the analytic ground-truth
boundary, identical to the one used for scoring.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adapters as ad
import regions as rg
import scene_gen as sg

COLOURS = ["#3b6fb6", "#d1663a", "#4f9d69", "#8a63a8", "#c9a227", "#5f6b7a"]


def _match_to_truth(scene, labels):
    """Map predicted ids onto ground-truth ids by best IoU.

    Without this the colours differ between panels (clusters are numbered
    arbitrarily), and the reader cannot tell a correct partition from a
    permuted one.
    """
    labels = np.asarray(labels)
    grid = rg.Grid(step=2.0)
    gt = {i: rg.polygon_mask(p, grid) for i, p in enumerate(scene["polys"], 1)}
    pm = rg.label_masks(rg.rasterize_labels(scene["X"], labels, grid))
    # rasterize_labels renumbers the surviving labels to 1..K, so predicted ids
    # have to be translated back before the mapping can be looked up
    uniq = sorted({int(v) for v in labels if v >= 0})
    ren = {u: i + 1 for i, u in enumerate(uniq)}
    mapping = {}
    if pm:
        gk, pk, M = rg.iou_matrix(gt, pm)
        rows, cols = rg.linear_sum_assignment(-M)
        for i, j in zip(rows, cols):
            if M[i, j] > 0:
                mapping[pk[j]] = gk[i]
    return {u: mapping.get(ren[u], 0) for u in uniq}


def draw(ax, scene, labels=None, title=""):
    if labels is None:
        col = [COLOURS[0] if v == 1 else COLOURS[1] for v in scene["y"]]
    else:
        labels = np.asarray(labels)
        mapping = _match_to_truth(scene, labels)
        col = ["#bbbbbb" if v < 0 else
               (COLOURS[(mapping[int(v)] - 1) % len(COLOURS)] if mapping.get(int(v))
                else "#8f8f8f") for v in labels]
    ax.scatter(scene["X"][:, 0], scene["X"][:, 1], s=2.2, c=col, linewidths=0)
    for p in scene["polys"]:
        geoms = [p] if p.geom_type == "Polygon" else list(p.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, "k--", lw=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="*",
                    default=["IZGC-Full", "K-means++", "DBSCAN", "BANG"])
    ap.add_argument("--out", default="fig_boundary")
    args = ap.parse_args()

    params = json.load(open("params_selected.json", encoding="utf-8"))
    alphas = json.load(open("alpha_selected.json", encoding="utf-8"))
    scenes = list(sg.SCENE_NAMES)
    cols = 1 + len(args.methods)

    fig, axes = plt.subplots(len(scenes), cols,
                             figsize=(2.15 * cols, 2.25 * len(scenes)))
    for i, nm in enumerate(scenes):
        scene = sg.build(nm, args.seed)
        draw(axes[i, 0], scene, title=f"{nm}\nground truth")
        for j, m in enumerate(args.methods, start=1):
            p = dict(params[f"{nm}|{m}"]["params"])
            if m in ad.POINT_METHODS:
                p["alpha"] = alphas[nm]["alpha"]
            try:
                labels, _ = ad.run(m, scene["X"], scene["k"], p, scene=scene, seed=args.seed)
            except Exception:
                labels = np.zeros(len(scene["X"]), int)
            draw(axes[i, j], scene, labels, title=m if i == 0 else "")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
