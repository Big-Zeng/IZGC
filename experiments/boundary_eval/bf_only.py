"""Boundary F1 (BF) only - run a subset and report nothing but boundary scores.

Why a separate entry point: `run_boundary.py` is a three-stage campaign
(parameter search -> alpha selection -> full evaluation of 7 methods x 6
scenes x 20 seeds).  When only the boundary score is needed - a single scene,
a single method, a different tolerance, or a sanity check - that machinery is
in the way.  This script reuses the parameters already selected in
`params_selected.json` / `alpha_selected.json`, so no tuning happens and the
numbers stay comparable with `results_boundary.csv`.

Usage
-----
    # all six scenes, all seven methods, 20 seeds
    python bf_only.py

    # one method on one scene
    python bf_only.py --scene gap_4 --methods IZGC-Full

    # three seeds only, extra tolerance of 2.0 units
    python bf_only.py --scene nonconvex_c --methods IZGC-Full K-means++ \
        --seeds 3 --tols 1.0 2.0

    # recompute the BF table from an existing results file (no clustering at all)
    python bf_only.py --from-csv results_boundary.csv

Output
------
    bf_runs.csv     one row per (scene, method, seed), BF at every tolerance
    bf_table.md     mean +- sd per (scene, method, tolerance)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GRID_STEP = 1.0
DEFAULT_TOLS = (1.0, 1.5, 2.5)
MAIN_TOL = 1.5          # the tolerance reported in the paper


# --------------------------------------------------------------------------- #
# one (scene, method, seed) run - module level so Windows spawn can import it
# --------------------------------------------------------------------------- #
def _worker(task):
    import adapters as ad
    import metrics as mt
    import regions as rg
    import scene_gen as sg

    scene_name, seed, method, params, tols = task
    scene = sg.build(scene_name, seed)
    grid = rg.Grid(step=GRID_STEP)

    rec = dict(scene=scene_name, family=scene["family"], method=method, seed=seed)
    try:
        labels, polys = ad.run(method, scene["X"], scene["k"], params,
                               scene=scene, seed=seed, polygonise=True)
        gt_pts = np.vstack([rg.boundary_points(p, grid.step) for p in scene["polys"]])
        pred_pts = (np.vstack([rg.boundary_points(p, grid.step) for p in polys])
                    if polys else np.empty((0, 2)))
        for t in tols:
            f1, prec, rec_ = mt.boundary_scores(gt_pts, pred_pts, t)
            rec[f"BF@{t:g}"] = f1
            if abs(t - MAIN_TOL) < 1e-9:
                rec["BPrec"], rec["BRec"] = prec, rec_
        rec.setdefault("BPrec", 0.0)
        rec.setdefault("BRec", 0.0)
        rec["n_gt_boundary"] = len(gt_pts)
        rec["n_pred_boundary"] = len(pred_pts)
    except Exception as exc:
        for t in tols:
            rec[f"BF@{t:g}"] = 0.0
        rec.update(BPrec=0.0, BRec=0.0, n_gt_boundary=0, n_pred_boundary=0,
                   error=str(exc)[:120])
    return rec


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _table(df, tols, label="scene"):
    cols = [f"BF@{t:g}" for t in tols]
    lines = []
    head = f"| {label} | method | n | " + " | ".join(cols) + " | BPrec | BRec |"
    lines.append(head)
    lines.append("|---|---|---|" + "---|" * (len(cols) + 2))
    for (key, method), g in df.groupby([label, "method"], sort=False):
        cells = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1) if len(g) > 1 else 0:.3f}"
                 for c in cols]
        lines.append(f"| {key} | {method} | {len(g)} | " + " | ".join(cells)
                     + f" | {g['BPrec'].mean():.3f} | {g['BRec'].mean():.3f} |")
    # overall row per method
    lines.append("")
    lines.append(f"| all | method | n | " + " | ".join(cols) + " | BPrec | BRec |")
    lines.append("|---|---|---|" + "---|" * (len(cols) + 2))
    for method, g in df.groupby("method", sort=False):
        cells = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1) if len(g) > 1 else 0:.3f}"
                 for c in cols]
        lines.append(f"| all | {method} | {len(g)} | " + " | ".join(cells)
                     + f" | {g['BPrec'].mean():.3f} | {g['BRec'].mean():.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Boundary F1 only")
    ap.add_argument("--scene", "--scenes", dest="scenes", nargs="*", default=None)
    ap.add_argument("--method", "--methods", dest="methods", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=20, help="number of test seeds")
    ap.add_argument("--seed-list", nargs="*", type=int, default=None,
                    help="explicit seeds, overrides --seeds")
    ap.add_argument("--tols", nargs="*", type=float, default=list(DEFAULT_TOLS),
                    help="boundary tolerances in scene units")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--params", default="params_selected.json")
    ap.add_argument("--alphas", default="alpha_selected.json")
    ap.add_argument("--out", default="bf_runs.csv")
    ap.add_argument("--table", default="bf_table.md")
    ap.add_argument("--from-csv", default=None,
                    help="aggregate an existing results csv instead of re-running")
    args = ap.parse_args()

    import pandas as pd

    tols = args.tols

    # ---- offline path: no clustering, just re-aggregate -------------------- #
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        for t in tols:
            if f"BF@{t:g}" not in df.columns:
                ap.error(f"{args.from_csv} has no column BF@{t:g}; "
                         f"available: {[c for c in df.columns if c.startswith('BF')]}")
        if args.scenes:
            df = df[df["scene"].isin(args.scenes)]
        if args.methods:
            df = df[df["method"].isin(args.methods)]
        table = _table(df, tols)
        open(args.table, "w", encoding="utf-8").write(table + "\n")
        print(table)
        print(f"\n[from-csv] {len(df)} rows -> {args.table}")
        return

    # ---- run path: reuse the selected parameters -------------------------- #
    import adapters as ad
    import scene_gen as sg

    scenes = args.scenes or list(sg.SCENE_NAMES)
    methods = args.methods or list(ad.METHODS)
    seeds = args.seed_list if args.seed_list else list(range(args.seeds))

    params_all = json.load(open(args.params, encoding="utf-8"))
    alphas = json.load(open(args.alphas, encoding="utf-8"))

    tasks, skipped = [], []
    for sc in scenes:
        alpha = alphas.get(sc, {}).get("alpha")
        for m in methods:
            key = f"{sc}|{m}"
            if key not in params_all:
                skipped.append(key)
                continue
            p = dict(params_all[key]["params"])
            if m in ad.POINT_METHODS:
                if alpha is None:
                    skipped.append(key)
                    continue
                p["alpha"] = alpha
            for sd in seeds:
                tasks.append((sc, sd, m, p, tols))

    if skipped:
        print(f"no selected parameters, skipped: {', '.join(skipped)}", flush=True)
    if not tasks:
        print("nothing to run", flush=True)
        return

    print(f"running {len(tasks)} jobs "
          f"({len(scenes)} scene x {len(methods)} method x {len(seeds)} seeds), "
          f"tolerances {tols}", flush=True)

    import multiprocessing as mp

    t0 = time.time()
    rows, done = [], 0
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        for rec in pool.imap_unordered(_worker, tasks, chunksize=1):
            rows.append(rec)
            done += 1
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  [{time.time() - t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows).sort_values(["scene", "method", "seed"])
    df.to_csv(args.out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df.columns else 0
    print(f"\nwrote {args.out}  ({len(df)} rows, {n_err} failed runs)")

    table = _table(df, tols)
    open(args.table, "w", encoding="utf-8").write(table + "\n")
    print("\n" + table)
    print(f"\nwrote {args.table}")


if __name__ == "__main__":
    main()
