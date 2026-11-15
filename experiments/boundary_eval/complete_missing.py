"""Fill in runs that are missing from results_boundary.csv.

`eval_scene.py` streams results through `imap_unordered`; if one task raises
outside its own try block the stream ends and the scene is left partially
written.  This script finds the (scene, seed, method) combinations that are
absent, evaluates just those, and appends them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GRID_STEP = 1.0


def worker(task):
    """Evaluate one run; never raises, so the result stream cannot break."""
    from sklearn.metrics import adjusted_rand_score

    import adapters as ad
    import metrics as mt
    import regions as rg
    import scene_gen as sg

    scene_name, seed, method, params, alpha = task
    try:
        scene = sg.build(scene_name, seed)
        grid = rg.Grid(step=GRID_STEP)
        labels, polys = ad.run(method, scene["X"], scene["k"], params,
                               scene=scene, seed=seed, polygonise=True)
        rec = mt.evaluate(scene, polys, labels, grid)
        rec["ARI"] = float(adjusted_rand_score(scene["y"], labels))
        rec.update(scene=scene_name, family=scene["family"], seed=seed,
                   method=method, params=json.dumps(params, sort_keys=True),
                   alpha=alpha)
        return rec
    except Exception as exc:
        rec = dict(scene=scene_name, family=scene_name.split("_")[0].rstrip("0123456789"),
                   seed=seed, method=method,
                   params=json.dumps(params, sort_keys=True), alpha=alpha,
                   BF=0.0, BPrec=0.0, BRec=0.0, HD95pct=100.0, IoU_pen=0.0,
                   IoU_match=0.0, k_gt=2, k_pred=0, k_matched=0, extra=0,
                   missed=2, leak=0.0, noise_frac=1.0, n_pred_boundary=0,
                   ARI=0.0, error=str(exc)[:160])
        for t in (1.0, 1.5, 2.5):
            rec[f"BF@{t:g}"] = 0.0
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_boundary.csv")
    ap.add_argument("--params", default="params_selected.json")
    ap.add_argument("--alphas", default="alpha_selected.json")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--budget", type=float, default=900.0)
    args = ap.parse_args()

    import adapters as ad

    df = pd.read_csv(args.results)
    params = json.load(open(args.params, encoding="utf-8"))
    alphas = json.load(open(args.alphas, encoding="utf-8"))

    have = set(zip(df.scene, df.seed.astype(int), df.method))
    methods = sorted(df.method.unique())
    scenes = sorted(df.scene.unique())

    tasks = []
    for sc in scenes:
        for seed in range(args.seeds):
            for m in methods:
                if (sc, seed, m) in have:
                    continue
                p = dict(params[f"{sc}|{m}"]["params"])
                alpha = alphas[sc]["alpha"]
                if m in ad.POINT_METHODS:
                    p["alpha"] = alpha
                tasks.append((sc, seed, m, p, alpha if m in ad.POINT_METHODS else 0.0))

    print(f"missing runs: {len(tasks)}")
    if not tasks:
        return

    rows, t0 = [], time.time()
    pool = mp.Pool(args.workers)

    def collect():
        for rec in pool.imap_unordered(worker, tasks, 1):
            rows.append(rec)
            if len(rows) % 10 == 0:
                print(f"  {len(rows)}/{len(tasks)}  [{time.time() - t0:.0f}s]", flush=True)

    th = threading.Thread(target=collect, daemon=True)
    th.start()
    th.join(args.budget)
    pool.terminate()
    pool.join()
    print(f"recovered {len(rows)}/{len(tasks)} in {time.time() - t0:.0f}s")

    add = pd.DataFrame(rows)
    out = pd.concat([df, add], ignore_index=True).sort_values(["scene", "method", "seed"])
    out.to_csv(args.results, index=False)
    print("coverage now:")
    print(out.pivot_table(index="scene", columns="method", values="seed",
                          aggfunc="count").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
