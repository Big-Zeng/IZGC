"""Rebuild `params_selected.json` / `alpha_selected.json` from the result table.

The two caches only hold the settings that `run_boundary.py` stage 3 needs:
for every (scene, method) the selected parameter dict, and for every scene the
alpha of the shared polygonisation.  Both are also recorded row-wise in
`results_boundary.csv` (`params` and `alpha` columns), because every evaluated
run stored the settings it used.  This script therefore restores the caches
without re-running any search - useful after the json files are deleted or when
they are regenerated on another machine.

    python restore_cache.py --results results_boundary.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_boundary.csv")
    ap.add_argument("--params-out", default="params_selected.json")
    ap.add_argument("--alpha-out", default="alpha_selected.json")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.results)
    for col in ("scene", "method", "params"):
        if col not in df.columns:
            ap.error(f"{args.results} has no column '{col}'")

    # tuning metadata, if the per-group dumps survived
    meta = {}
    for path in glob.glob(os.path.join(HERE, "_grp_*.json")):
        try:
            payload = json.load(open(path, encoding="utf-8"))
            key = f"{payload['scene']}|{payload['method']}"
            if payload.get("params"):
                meta[key] = payload
        except Exception:
            continue

    params_out = {}
    for (scene, method), g in df.groupby(["scene", "method"], sort=False):
        p = json.loads(g["params"].iloc[0])
        m = meta.get(f"{scene}|{method}", {})
        params_out[f"{scene}|{method}"] = dict(
            params=p,
            val_ARI=m.get("val_ARI"),
            val_k=m.get("val_k"),
            n_grid=m.get("n_grid", 0),
            n_done=m.get("n_done", 0),
            note="rebuilt from " + os.path.basename(args.results),
        )
    json.dump(params_out, open(args.params_out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {args.params_out}  ({len(params_out)} entries)")

    alpha_out = {}
    if "alpha" in df.columns:
        for scene, g in df.groupby("scene", sort=False):
            vals = g.loc[g["alpha"] > 0, "alpha"]
            alpha_out[scene] = dict(
                alpha=round(float(vals.iloc[0]), 4) if len(vals) else None,
                val_IoU=None,
            )
        json.dump(alpha_out, open(args.alpha_out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {args.alpha_out}  ({len(alpha_out)} scenes)")
    else:
        print("no 'alpha' column, alpha cache left untouched")


if __name__ == "__main__":
    main()
