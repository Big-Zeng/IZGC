"""Tune ONE (scene, method) pair in a separate process.

Two stages, one objective per stage
-----------------------------------
Stage 1 sweeps the published grid and keeps the ARI, i.e. the paper's own
selection metric.  Stage 2 only looks at the configurations that tie on the
*best* ARI and breaks the tie by boundary F1 on the same validation seed.
Ties are broken a second time by grid step (median of the tied set), so the
choice is deterministic: the same grid always yields the same parameters.

Tie-breaking by BF is legitimate only because ARI is saturated - many steps
give exactly the same partition quality - and because it runs on a validation
seed that is never used for the reported scores.  Every method goes through the
same rule, so no method is favoured.

The parent runs this file through `subprocess.run(timeout=...)`, so a
pathological configuration can never freeze the whole experiment: the group is
killed and the parent moves on.  Inside, a watchdog thread terminates the worker
pool once the wall-clock budget is spent, keeping the best configuration found
so far.
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
BF_TOL = 1.5
ARI_DECIMALS = 6       # two ARI values count as tied at this precision


def _scene_alpha(scene_name):
    path = os.path.join(HERE, "alpha_selected.json")
    try:
        return float(json.load(open(path, encoding="utf-8"))[scene_name]["alpha"])
    except Exception:
        return 0.05


def worker_ari(task):
    """Stage 1: one grid point, scored by ARI only (no polygonisation)."""
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


def worker_bf(task):
    """Stage 2: one tied grid point, scored by boundary F1 only."""
    import numpy as np

    import adapters as ad
    import metrics as mt
    import regions as rg
    import scene_gen as sg

    scene_name, seed, method, params = task
    scene = sg.build(scene_name, seed)
    grid = rg.Grid(step=GRID_STEP)
    try:
        labels, polys = ad.run(method, scene["X"], scene["k"], params, scene=scene,
                               seed=seed, polygonise=True)
        gt_pts = np.vstack([rg.boundary_points(p, grid.step) for p in scene["polys"]])
        pred_pts = (np.vstack([rg.boundary_points(p, grid.step) for p in polys])
                    if polys else np.empty((0, 2)))
        bf, prec, rec = mt.boundary_scores(gt_pts, pred_pts, BF_TOL)
        k_hat = len(set(int(v) for v in labels) - {-1})
        return (params, float(bf), k_hat)
    except Exception:
        return (params, -1.0, 0)


def _sweep(fn, tasks, workers, budget, serial=False):
    results: list = []
    t0 = time.time()
    if serial:
        # MATLAB methods keep one engine per process, so they cannot be
        # scattered over a worker pool
        for task in tasks:
            results.append(fn(task))
            if time.time() - t0 > budget:
                break
        return results, time.time() - t0 > budget, time.time() - t0

    def collect(pool):
        for r in pool.imap_unordered(fn, tasks, chunksize=1):
            results.append(r)

    pool = mp.Pool(min(workers, max(1, len(tasks))))
    th = threading.Thread(target=collect, args=(pool,), daemon=True)
    th.start()
    th.join(budget)
    timed_out = th.is_alive()
    pool.terminate()          # immediate, never waits for a stuck task
    pool.join()
    return results, timed_out, time.time() - t0


def _step(params):
    return params.get("grid_size", params.get("levels", params.get("eps",
                   params.get("k", 0))))


BF_MAX_RUNS = 400        # above this many ties, score BF coarse-then-refine
COARSE_MULT = 100        # ... on multiples of 0.01 when the grid step is 0.001
REFINE_HALF = 0.02       # ... and refine +-0.02 around the coarse winner


def _is_coarse(params):
    s = _step(params)
    return abs(round(s * COARSE_MULT) - s * COARSE_MULT) < 1e-6


def _bf_subset(ties):
    """Which tied configurations get a boundary score.

    With a 0.001 grid the ARI plateau of IZGC holds roughly 1200 steps, all
    with the same partition quality.  Scoring every one of them costs an hour
    per scene for a difference in the fourth decimal, so the plateau is scored
    every 0.01 and then refined at 0.001 inside +-0.02 of the coarse peak.
    Small tie sets (every point method) are scored exhaustively.
    """
    if len(ties) <= BF_MAX_RUNS:
        return list(ties), "full"
    coarse = [p for p in ties if _is_coarse(p)] or list(ties)
    return coarse, "coarse"


def _refine(ties, coarse_winner, already):
    lo, hi = _step(coarse_winner) - REFINE_HALF, _step(coarse_winner) + REFINE_HALF
    out = [p for p in ties if lo - 1e-9 <= _step(p) <= hi + 1e-9
           and json.dumps(p, sort_keys=True) not in already]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import adapters as ad

    grid = ad.param_grid(args.method)
    alpha = _scene_alpha(args.scene) if args.method in ad.POINT_METHODS else None

    # ---------------- stage 1: ARI over the whole grid ------------------- #
    tasks = [(args.scene, args.seed, args.method, p) for p in grid]
    ari_results, timed_out1, sec1 = _sweep(worker_ari, tasks, args.workers,
                                           max(60.0, args.budget * 0.7),
                                           serial=args.method in ad.MATLAB_METHODS)

    # start at -1.0, the sentinel a failed run returns, so that a method which
    # fails everywhere keeps params=None and the caller can fall back
    best_ari, best_k, best_params = -1.0, 0, None
    for params, ari, k_hat in ari_results:
        if ari > best_ari:
            best_ari, best_k, best_params = ari, k_hat, params

    # ---------------- stage 2: BF only among the ARI ties ---------------- #
    n_tied, bf_best, sec2, timed_out2 = 0, None, 0.0, False
    bf_mode, n_bf = "none", 0
    ties = []
    if best_ari > 0:
        ties = [dict(p) for p, ari, _ in ari_results
                if round(ari, ARI_DECIMALS) == round(best_ari, ARI_DECIMALS)]
        ties.sort(key=_step)
        n_tied = len(ties)
        if n_tied > 1:
            def _key(p):
                return json.dumps({k: v for k, v in p.items() if k != "alpha"},
                                  sort_keys=True)

            def _eval(subset):
                tasks_ = []
                for p in subset:
                    q = dict(p)
                    if alpha is not None:
                        q["alpha"] = alpha
                    tasks_.append((args.scene, args.seed, args.method, q))
                return _sweep(worker_bf, tasks_, args.workers,
                              max(60.0, args.budget * 0.3),
                              serial=args.method in ad.MATLAB_METHODS)

            subset, bf_mode = _bf_subset(ties)
            bf_results, timed_out2, sec2 = _eval(subset)
            n_bf = len(bf_results)
            scored = {_key(p): bf for p, bf, _ in bf_results}
            top = max(scored.values()) if scored else -1.0

            # second pass, only when the plateau was scored coarsely
            if bf_mode == "coarse" and scored:
                winners0 = [p for p in subset if scored.get(_key(p), -2.0) == top]
                winners0.sort(key=_step)
                mid = winners0[len(winners0) // 2]
                more = _refine(ties, mid, set(scored))
                if more:
                    bf_results2, to2, s2 = _eval(more)
                    timed_out2 = timed_out2 or to2
                    sec2 += s2
                    n_bf += len(bf_results2)
                    bf_mode = "coarse+refine"
                    bf_results += bf_results2
                    scored = {_key(p): bf for p, bf, _ in bf_results}
                    top = max(scored.values())

            winners = [p for p in ties if scored.get(_key(p), -2.0) == top]
            winners.sort(key=_step)
            if winners:                     # median step -> deterministic
                best_params = winners[len(winners) // 2]
                best_k = next((k for p, _, k in bf_results
                               if _key(p) == _key(best_params)), best_k)
                bf_best = round(float(top), 4)
    elif ties:
        best_params = ties[0]

    payload = dict(scene=args.scene, method=args.method, params=best_params,
                   val_ARI=round(float(best_ari), 4), val_k=int(best_k),
                   n_grid=len(grid), n_done=len(ari_results),
                   timed_out=bool(timed_out1 or timed_out2),
                   n_tied=int(n_tied), val_BF=bf_best, alpha=alpha,
                   bf_mode=bf_mode, n_bf=int(n_bf),
                   seconds=round(sec1 + sec2, 1),
                   stage1_seconds=round(sec1, 1), stage2_seconds=round(sec2, 1))
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"{args.scene} {args.method} best={best_params} ARI={best_ari:.3f} "
          f"BF={bf_best} tied={n_tied} ({len(ari_results)}/{len(grid)} configs, "
          f"{payload['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
