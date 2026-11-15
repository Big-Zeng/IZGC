"""Summarise the boundary experiment: tables, tests, markdown report.

Outputs
-------
table_by_scene.csv   mean per (scene, method)
table_overall.csv    mean and sd per method over all scenes and seeds
table_tests.csv      Friedman test + paired Wilcoxon (Holm) against IZGC-Full
summary_boundary.md  the same tables in markdown, ready to paste into notes
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

METRICS = ["BF", "BPrec", "BRec", "HD95pct", "IoU_pen", "ARI", "k_pred"]
ORDER = ["IZGC-Full", "IZGC-NoZP", "K-means++", "GMM", "DBSCAN", "CLIQUE",
         "BANG", "DPC-FWSN"]


def load(path="results_boundary.csv"):
    df = pd.read_csv(path)
    if "error" in df.columns:
        bad = df[df["error"].notna()]
        if len(bad):
            print(f"  note: {len(bad)} failing runs counted as zero scores")
            print(bad.groupby(["method"]).size().to_string())
    return df


def merge_inputs(paths):
    """Concatenate result files, later files replacing earlier (method, scene)."""
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["_ord"] = range(len(df))
    df = (df.sort_values("_ord")
            .drop_duplicates(subset=["scene", "method", "seed"], keep="last")
            .drop(columns="_ord"))
    return df.sort_values(["scene", "method", "seed"]).reset_index(drop=True)


def table_by_scene(df):
    t = df.groupby(["scene", "method"])[METRICS].mean().round(3)
    return t.reset_index()


def table_overall(df):
    rows = []
    for m in ORDER:
        d = df[df.method == m]
        if not len(d):
            continue
        row = dict(method=m)
        for c in ["BF", "BPrec", "BRec", "HD95pct", "IoU_pen", "ARI"]:
            row[c] = round(float(d[c].mean()), 3)
            row[f"{c}_sd"] = round(float(d[c].std(ddof=1)), 3)
        # BF_alpha = boundary F1 under the shared alpha-shape representation.
        # Point-label methods are already scored that way, so their BF is the
        # same number; only IZGC has a second value.
        if "BF_alpha" in d.columns:
            row["BF_alpha"] = round(float(d["BF_alpha"].fillna(d["BF"]).mean()), 3)
        else:
            row["BF_alpha"] = row["BF"]
        row["k_pred"] = round(float(d["k_pred"].mean()), 2)
        row["extra"] = round(float(d["extra"].mean()), 2)
        row["missed"] = round(float(d["missed"].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def tests(df, ref="IZGC-Full", metric="BF"):
    """Friedman over the (scene, seed) blocks, then Wilcoxon against the ref."""
    pivot = df.pivot_table(index=["scene", "seed"], columns="method", values=metric)
    cols = [c for c in ORDER if c in pivot.columns]
    pivot = pivot[cols].dropna()
    if pivot.empty or len(cols) < 3:
        return pd.DataFrame()
    stat, p = friedmanchisquare(*[pivot[c].values for c in cols])
    out = [dict(metric=metric, comparison="Friedman (all methods)", statistic=round(stat, 3),
                p_value=p, p_holm=np.nan, note=f"{len(cols)} methods, {len(pivot)} blocks")]
    others = [c for c in cols if c != ref]
    raw = []
    for c in others:
        try:
            s, pv = wilcoxon(pivot[ref].values, pivot[c].values)
        except ValueError:
            s, pv = np.nan, 1.0
        raw.append((c, float(s), float(pv)))
    # Holm correction over this family of comparisons
    order = sorted(range(len(raw)), key=lambda i: raw[i][2])
    m = len(raw)
    adj = [1.0] * m
    for rank, i in enumerate(order):
        adj[i] = min(1.0, raw[i][2] * (m - rank))
    for i, (c, s, pv) in enumerate(raw):
        out.append(dict(metric=metric, comparison=f"{ref} vs {c}", statistic=round(s, 3),
                        p_value=round(pv, 4), p_holm=round(adj[i], 4), note="paired Wilcoxon, Holm"))
    return pd.DataFrame(out)


def to_markdown(by_scene, overall, test_tables):
    lines = ["# Boundary-quality experiment - results", ""]
    lines += ["## Overall (mean over 6 scenes x 20 seeds)", "",
              "BF = each method in its own representation (IZGC contours, alpha shapes for the rest).",
              "BF_alpha = every method after the shared alpha-shape polygonisation, so the "
              "comparison holds the boundary representation fixed.", "",
              "| method | BF | BF_alpha | BPrec | BRec | HD95 (%diag) | IoU | ARI | k_pred | extra | missed |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in overall.iterrows():
        lines.append(f"| {r.method} | {r.BF:.3f} | {r.BF_alpha:.3f} | {r.BPrec:.3f} | {r.BRec:.3f} | "
                     f"{r.HD95pct:.2f} | {r.IoU_pen:.3f} | {r.ARI:.3f} | {r.k_pred:.2f} | "
                     f"{r.extra:.2f} | {r.missed:.2f} |")
    lines += ["", "## Per scene (BF / IoU)", "",
              "| scene | " + " | ".join(ORDER) + " |",
              "|---|" + "---|" * len(ORDER)]
    for sc in sorted(by_scene.scene.unique()):
        d = by_scene[by_scene.scene == sc].set_index("method")
        cells = [f"{d.loc[m, 'BF']:.3f} / {d.loc[m, 'IoU_pen']:.3f}" if m in d.index else "-"
                 for m in ORDER]
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    lines += ["", "## Statistical tests", ""]
    for t in test_tables:
        if len(t):
            lines.append(f"### metric: {t.metric.iloc[0]}")
            lines.append("")
            lines.append("| " + " | ".join(t.columns) + " |")
            lines.append("|" + "---|" * len(t.columns))
            for _, row in t.iterrows():
                lines.append("| " + " | ".join(
                    "" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                    for v in row.values) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=["results_boundary.csv"],
                    help="one or more result csv files; later files win")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df = merge_inputs(args.input) if len(args.input) > 1 else load(args.input[0])
    by_scene = table_by_scene(df)
    overall = table_overall(df)
    tests_all = [tests(df, metric=m) for m in ("BF", "IoU_pen", "HD95pct")]
    tests_df = pd.concat([t for t in tests_all if len(t)], ignore_index=True)

    by_scene.to_csv(f"table_by_scene{args.tag}.csv", index=False)
    overall.to_csv(f"table_overall{args.tag}.csv", index=False)
    tests_df.to_csv(f"table_tests{args.tag}.csv", index=False)
    open(f"summary_boundary{args.tag}.md", "w", encoding="utf-8").write(
        to_markdown(by_scene, overall, tests_all))

    pd.set_option("display.width", 240)
    print("\n== overall ==\n", overall.to_string(index=False))
    print("\n== per scene: BF ==\n",
          by_scene.pivot(index="scene", columns="method", values="BF").round(3).to_string())
    print("\n== tests ==\n", tests_df.to_string(index=False))


if __name__ == "__main__":
    main()
