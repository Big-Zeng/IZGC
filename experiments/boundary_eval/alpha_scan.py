"""Re-run the alpha-shape selection on the validation seed and record val_IoU.

`alpha_selected.json` currently stores alpha=0.05 for all six scenes with
val_IoU=null, i.e. the file is not the output of search_alpha().  This script
reproduces the documented procedure (validation seed 9001, mean IoU_pen of the
five point-label methods) over a slightly wider grid so that the manuscript
statement has a recorded basis, and reports the resulting BF as well.

Usage:
    python alpha_scan.py
"""
import json
import os

import numpy as np
import pandas as pd

import adapters as ad
import metrics as mt
import regions as rg
import scene_gen as sg

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_SEED = 9001
ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.35]
METHODS = ["K-means++", "GMM", "DBSCAN", "CLIQUE", "BANG"]
SCENES = ["nonconvex_c", "nonconvex_u", "gap_4", "gap_2", "density_3", "density_8"]


def main():
    params_old = json.load(open(os.path.join(HERE, "params_selected.json"), encoding="utf-8"))
    params_gap = json.load(open(os.path.join(HERE, "_base_gapfix_params.json"), encoding="utf-8"))
    grid = rg.Grid(step=1.0)
    rows = []

    for sc in SCENES:
        scene = sg.build(sc, VAL_SEED)
        for m in METHODS:
            key = f"{sc}|{m}"
            src = params_gap if key in params_gap else params_old
            if key not in src:
                print(f"  skip {key}: no parameters")
                continue
            p = dict(src[key]["params"])
            labels, _ = ad.run_point_method(m, scene["X"], scene["k"], p,
                                            alpha=0.05, seed=VAL_SEED, polygonise=False)
            for a in ALPHAS:
                polys = ad._polygonise(scene["X"], labels, a)
                rec = mt.evaluate(scene, polys, labels, grid)
                rows.append(dict(scene=sc, method=m, alpha=a,
                                 IoU=rec["IoU_pen"], BF=rec["BF"]))
        print(f"  {sc} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "alpha_scan.csv"), index=False)

    agg = df.groupby(["scene", "alpha"])[["IoU", "BF"]].mean().reset_index()
    print("\n=== per scene, mean over the five point methods (validation seed 9001) ===")
    pivot = agg.pivot(index="scene", columns="alpha", values="IoU").round(4)
    print(pivot.to_string())
    print("\n=== mean BF ===")
    print(agg.pivot(index="scene", columns="alpha", values="BF").round(4).to_string())

    best = agg.loc[agg.groupby("scene")["IoU"].idxmax()]
    print("\n=== best alpha by mean IoU ===")
    print(best.to_string(index=False))
    print("\noverall mean IoU per alpha:")
    print(agg.groupby("alpha")["IoU"].mean().round(4).to_string())

    out = {r.scene: dict(alpha=float(r.alpha), val_IoU=round(float(r.IoU), 4))
           for r in best.itertuples()}
    json.dump(out, open(os.path.join(HERE, "alpha_scan_selected.json"), "w",
                        encoding="utf-8"), indent=1)
    print("\nwrote alpha_scan.csv / alpha_scan_selected.json")


if __name__ == "__main__":
    main()
