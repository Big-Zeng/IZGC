"""Boundary-quality experiment: parameter search, evaluation, tables.

Protocol
--------
Stage 1  parameter search, per scene and per method, on a validation seed.
         Grid = supplementary Table S4 ranges with the step sizes of
         `algorithm_comparison/run_all.py`; objective = ARI, i.e. the paper's
         own selection rule.  ARI never looks at a boundary, so the boundary
         scores below cannot have been tuned.
Stage 2  alpha of the shared alpha-shape polygonisation, chosen per scene on
         the same validation seed by area agreement (IoU), so the point-label
         methods obtain an outline of their own region rather than a crippled
         one.
Stage 3  evaluation of every method on `--seeds` fresh test seeds with those
         fixed settings: boundary F1, HD95, IoU, leakage and cluster counts.

Usage
-----
    python run_boundary.py --seeds 20 --workers 10
    python run_boundary.py --scenes gap_2 --methods DBSCAN   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VAL_SEED = 9001          # never used for the reported scores
GRID_STEP = 1.0          # raster resolution shared by every method


# --------------------------------------------------------------------------- #
# workers (module level so that they survive the Windows spawn)
# --------------------------------------------------------------------------- #
def _tune_worker(task):
    from sklearn.metrics import adjusted_rand_score

    import adapters as ad
    import scene_gen as sg

    scene_name, seed, method, params = task
    scene = sg.build(scene_name, seed)
    try:
        labels, _ = ad.run(method, scene["X"], scene["k"], params, scene=scene,
                           seed=seed, polygonise=False)
        k_hat = len(set(int(v) for v in labels) - {-1})
        if k_hat < 2:
            return (params, -1.0, k_hat)
        return (params, float(adjusted_rand_score(scene["y"], labels)), k_hat)
    except Exception:
        return (params, -1.0, 0)


def _eval_worker(task):
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
    except Exception as exc:                       # a failing method scores 0
        rec = dict(BF=0.0, BPrec=0.0, BRec=0.0, HD95pct=100.0, IoU_pen=0.0,
                   IoU_match=0.0, k_gt=scene["k"], k_pred=0, k_matched=0,
                   extra=0, missed=scene["k"], leak=0.0, noise_frac=1.0,
                   n_pred_boundary=0, ARI=0.0, error=str(exc)[:120])
        for t in (1.0, 1.5, 2.5):
            rec[f"BF@{t:g}"] = 0.0
    rec.update(scene=scene_name, family=scene["family"], seed=seed, method=method,
               params=json.dumps(params, sort_keys=True), alpha=alpha)
    return rec


# --------------------------------------------------------------------------- #
# stage 1: parameter search
# --------------------------------------------------------------------------- #
FALLBACK = {
    "IZGC-Full": dict(grid_size=2.0, R=30.0, min_size=5, buffer=0),
    "IZGC-NoZP": dict(grid_size=2.0, R=30.0, min_size=5, buffer=0),
    "K-means++": dict(),
    "GMM": dict(covariance_type="full", n_init=10),
    "DBSCAN": dict(eps=5.0, min_samples=5),
    "CLIQUE": dict(intervals=20, thr=5),
    "BANG": dict(levels=11, thr=0.05),
    "DPC-FWSN": dict(k=11),
}


def _run_group(sc, m, workers, lock, cache, cache_path):
    """Tune one pair in a child process; returns (key, record)."""
    import adapters as ad
    import threading

    grid = ad.param_grid(m)
    # stage 1 (ARI sweep) plus stage 2 (BF over the ties) in one child process,
    # so the budget has to cover both.  A 0.001 Delta_g grid means 4501 runs.
    budget = min(5400.0, 900.0 + 2.0 * len(grid))
    tmp = f"_grp_{sc}_{m.replace('+', 'p')}.json"
    cmd = [sys.executable, os.path.join(HERE, "tune_group.py"),
           "--scene", sc, "--method", m, "--seed", str(VAL_SEED),
           "--workers", str(min(workers, max(2, len(grid)))),
           "--budget", f"{budget:.0f}", "--out", tmp]
    t0 = time.time()
    note = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=budget + 120, cwd=HERE)
        payload = json.load(open(tmp, encoding="utf-8"))
        params = payload["params"] or FALLBACK[m]
        if payload["timed_out"]:
            note = f"budget hit, {payload['n_done']}/{payload['n_grid']} configs"
        elif proc.returncode != 0:
            note = f"child rc={proc.returncode}"
    except Exception as exc:
        params = FALLBACK[m]
        payload = dict(val_ARI=None, val_k=None, n_done=0)
        note = f"fallback ({type(exc).__name__})"
    rec = dict(params=params, val_ARI=payload.get("val_ARI"),
               val_k=payload.get("val_k"), val_BF=payload.get("val_BF"),
               n_tied=payload.get("n_tied"), alpha=payload.get("alpha"),
               bf_mode=payload.get("bf_mode"), n_bf=payload.get("n_bf"),
               n_grid=len(grid), n_done=payload.get("n_done", 0), note=note)
    with lock:
        cache[f"{sc}|{m}"] = rec
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=1)
    print(f"  {sc:13s} {m:11s} ARI={payload.get('val_ARI')} k={payload.get('val_k')}  "
          f"{params}  [{time.time() - t0:.0f}s of {len(grid)}] {note}", flush=True)
    return rec


def search_parameters(scenes, methods, workers, cache_path, groups=3):
    """Tune every (scene, method) pair, `groups` of them concurrently.

    Each group runs in its own child process with a hard budget, so a
    pathological configuration can only cost that group; the parent never
    blocks on a worker pool.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))

    pending = [(sc, m) for sc in scenes for m in methods if f"{sc}|{m}" not in cache]
    for sc in scenes:
        for m in methods:
            if f"{sc}|{m}" in cache:
                print(f"  {sc:13s} {m:11s} cached (ARI={cache[f'{sc}|{m}']['val_ARI']})",
                      flush=True)
    if not pending:
        return cache

    per_group = max(2, workers // max(1, groups))
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=groups) as pool:
        list(pool.map(lambda a: _run_group(a[0], a[1], per_group, lock, cache, cache_path),
                      pending))
    return cache


# --------------------------------------------------------------------------- #
# stage 2: alpha of the shared alpha-shape polygonisation
# --------------------------------------------------------------------------- #
def search_alpha(scenes, cache, workers, path="alpha_selected.json"):
    chosen = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for sc in scenes:
        if sc in chosen:
            continue
        best = (None, -1.0)
        for a in (0.05, 0.10, 0.20, 0.35):
            vals = []
            for m in ("K-means++", "GMM", "DBSCAN", "CLIQUE", "BANG"):
                if f"{sc}|{m}" not in cache:
                    continue
                p = dict(cache[f"{sc}|{m}"]["params"])
                p["alpha"] = a
                try:
                    vals.append(_eval_worker((sc, VAL_SEED, m, p, a))["IoU_pen"])
                except Exception:
                    continue
            if vals and np.mean(vals) > best[1]:
                best = (a, float(np.mean(vals)))
        chosen[sc] = dict(alpha=best[0], val_IoU=round(best[1], 4))
        print(f"  alpha {sc:13s} = {best[0]}  (mean IoU on validation "
              f"{best[1]:.3f})", flush=True)
        json.dump(chosen, open(path, "w", encoding="utf-8"), indent=1)
    return {k: v["alpha"] for k, v in chosen.items()}


# --------------------------------------------------------------------------- #
# stage 3: evaluation
# --------------------------------------------------------------------------- #
def evaluate(scenes, methods, cache, alphas, seeds, workers, out,
             params_file="params_selected.json", alpha_file="alpha_selected.json"):
    """Evaluate one scene per child process (`eval_scene.py`), appending rows."""
    seeds = int(seeds)
    frames = []
    tag = os.path.splitext(os.path.basename(out))[0]   # keep parallel runs apart
    for sc in scenes:
        tmp = f"_{tag}_{sc}.jsonl"
        if os.path.exists(tmp):
            os.remove(tmp)
        cmd = [sys.executable, os.path.join(HERE, "eval_scene.py"),
               "--scene", sc, "--seeds", str(seeds), "--workers", str(workers),
               "--params", params_file, "--alphas", alpha_file,
               "--methods", *methods, "--out", tmp]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=3600, cwd=HERE)
            tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "no output"
            print(f"  {sc:13s} {tail}  [{time.time() - t0:.0f}s]", flush=True)
            if proc.returncode != 0:
                print("    stderr: " + proc.stderr.strip().splitlines()[-1], flush=True)
        except subprocess.TimeoutExpired:
            print(f"  {sc:13s} child exceeded 3600 s, keeping partial rows", flush=True)
        if not os.path.exists(tmp):
            continue
        rows = [json.loads(line) for line in open(tmp, encoding="utf-8") if line.strip()]
        frames.append(pd.DataFrame(rows))

    if not frames:
        print("  no rows produced", flush=True)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values(["scene", "method", "seed"])
    df.to_csv(out, index=False)
    print(f"  wrote {out}  ({len(df)} rows)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=20, help="number of test seeds")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results_boundary.csv")
    ap.add_argument("--params-file", default="params_selected.json",
                    help="where the selected parameters are cached")
    ap.add_argument("--alpha-file", default="alpha_selected.json")
    ap.add_argument("--groups", type=int, default=3,
                    help="how many (scene, method) pairs are tuned concurrently")
    ap.add_argument("--retune", action="store_true", help="ignore the caches")
    args = ap.parse_args()

    import adapters as ad
    import scene_gen as sg

    scenes = args.scenes or list(sg.SCENE_NAMES)
    methods = args.methods or list(ad.METHODS)
    cache_path = args.params_file
    if args.retune:
        for f in (cache_path, args.alpha_file):
            if os.path.exists(f):
                os.remove(f)

    print("== stage 1: parameter search (validation seed, objective ARI) ==", flush=True)
    cache = search_parameters(scenes, methods, args.workers, cache_path,
                              groups=args.groups)

    print("== stage 2: alpha-shape selection (validation seed, objective IoU) ==", flush=True)
    alphas = search_alpha(scenes, cache, args.workers, args.alpha_file)

    print("== stage 3: evaluation on test seeds ==", flush=True)
    evaluate(scenes, methods, cache, alphas, args.seeds, args.workers, args.out,
             args.params_file, args.alpha_file)
    sg.manifest()


if __name__ == "__main__":
    main()
