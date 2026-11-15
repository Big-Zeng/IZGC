"""Evaluate ONE scene over the test seeds in a separate process.

Same isolation idea as `tune_group.py`: the parent gives this file a wall-clock
timeout, so a single pathological run cannot freeze the campaign.  Results are
written as soon as they arrive, so a partial scene is still usable.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GRID_STEP = 1.0


def worker(task):
    from sklearn.metrics import adjusted_rand_score

    import adapters as ad
    import metrics as mt
    import regions as rg
    import scene_gen as sg

    scene_name, seed, method, params, alpha = task
    scene = sg.build(scene_name, seed)
    grid = rg.Grid(step=GRID_STEP)
    try:
        labels, polys = ad.run(method, scene["X"], scene["k"], params,
                               scene=scene, seed=seed, polygonise=True)
        rec = mt.evaluate(scene, polys, labels, grid)
        rec["ARI"] = float(adjusted_rand_score(scene["y"], labels))
        # IZGC in its own representation uses its contours; the same labels
        # polygonised with the shared alpha shape give the "same-formula" BF
        # that every point-label method is scored with.
        if method.startswith("IZGC") and alpha:
            alt = ad._polygonise(scene["X"], labels, alpha)
            r2 = mt.evaluate(scene, alt, labels, grid)
            for k_src in ("BF", "BPrec", "BRec", "HD95pct", "IoU_pen"):
                rec[f"{k_src}_alpha"] = r2[k_src]
            for t in (1.0, 1.5, 2.5):
                rec[f"BF@{t:g}_alpha"] = r2[f"BF@{t:g}"]
    except Exception as exc:
        rec = dict(BF=0.0, BPrec=0.0, BRec=0.0, HD95pct=100.0, IoU_pen=0.0,
                   IoU_match=0.0, k_gt=scene["k"], k_pred=0, k_matched=0,
                   extra=0, missed=scene["k"], leak=0.0, noise_frac=1.0,
                   n_pred_boundary=0, ARI=0.0, error=str(exc)[:120])
        for t in (1.0, 1.5, 2.5):
            rec[f"BF@{t:g}"] = 0.0
    rec.update(scene=scene_name, family=scene["family"], seed=seed, method=method,
               params=json.dumps(params, sort_keys=True), alpha=alpha)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--budget", type=float, default=1500.0)
    ap.add_argument("--params", required=True, help="params_selected.json")
    ap.add_argument("--alphas", required=True, help="alpha_selected.json")
    ap.add_argument("--methods", nargs="*", required=True)
    ap.add_argument("--out", required=True, help="one json line per run")
    args = ap.parse_args()

    import adapters as ad

    params = json.load(open(args.params, encoding="utf-8"))
    alphas = json.load(open(args.alphas, encoding="utf-8"))
    alpha = alphas[args.scene]["alpha"]

    tasks, serial_tasks = [], []
    for seed in range(args.seeds):
        for m in args.methods:
            p = dict(params[f"{args.scene}|{m}"]["params"])
            use_alpha = m in ad.POINT_METHODS
            if use_alpha:
                p["alpha"] = alpha
            item = (args.scene, seed, m, p,
                    alpha if (use_alpha or m.startswith("IZGC")) else 0.0)
            (serial_tasks if m in ad.MATLAB_METHODS else tasks).append(item)

    t0 = time.time()
    lock = threading.Lock()
    timed_out = False

    with open(args.out, "w", encoding="utf-8") as fh:
        if tasks:
            pool = mp.Pool(min(args.workers, len(tasks)))

            def collect():
                for rec in pool.imap_unordered(worker, tasks, chunksize=1):
                    with lock:
                        fh.write(json.dumps(rec, default=float) + "\n")
                        fh.flush()

            th = threading.Thread(target=collect, daemon=True)
            th.start()
            th.join(args.budget)
            timed_out = th.is_alive()
            pool.terminate()
            pool.join()

        # MATLAB methods keep one engine in this process; 20 seeds at ~6 s
        for item in serial_tasks:
            fh.write(json.dumps(worker(item), default=float) + "\n")
            fh.flush()

    n = sum(1 for _ in open(args.out, encoding="utf-8")) if os.path.exists(args.out) else 0
    print(f"{args.scene}: {n}/{len(tasks) + len(serial_tasks)} runs in "
          f"{time.time() - t0:.0f}s (timeout={timed_out})", flush=True)


if __name__ == "__main__":
    main()
