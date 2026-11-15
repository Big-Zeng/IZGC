"""Per-scene metric tables (6 scenes x 8 methods), unified alpha-shape boundary F1.

IZGC variants report the alpha-shape polygonisation (BF_alpha) so that every
method is scored by the same boundary formula; their native contour values are
printed in a supplementary table.

Usage:
    python tables_by_scene.py                 # writes tables_by_scene.md/.csv
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["nonconvex_c", "nonconvex_u", "gap_4", "gap_2", "density_3", "density_8"]
TITLES = {
    "nonconvex_c": "非凸边界 C 形（2 簇，凹口矩形）",
    "nonconvex_u": "非凸边界 U 形（2 簇，贯穿凹槽）",
    "gap_4": "狭窄间隔 4 单位（2 簇）",
    "gap_2": "狭窄间隔 2 单位（2 簇）",
    "density_3": "异质密度比 3:1（2 簇）",
    "density_8": "异质密度比 8:1（2 簇）",
}
ORDER = ["IZGC-Full", "IZGC-NoZP", "DPC-FWSN", "K-means++", "GMM", "DBSCAN", "CLIQUE", "BANG"]
NOTE = {
    "IZGC-Full": "IZGC，0.001 步长，ARI 并列内 BF 取优",
    "IZGC-NoZP": "去除零值虚拟点（消融）",
    "DPC-FWSN": "密度峰值 + 快速权重网络",
}


def load():
    frames = []

    iz = pd.read_csv(os.path.join(HERE, "results_izgc0001.csv"))
    iz = iz.copy()
    # unify: alpha-shape polygonisation for both IZGC variants
    for src, dst in [("BF_alpha", "BF"), ("BPrec_alpha", "BPrec"), ("BRec_alpha", "BRec"),
                     ("HD95pct_alpha", "HD95pct"), ("IoU_pen_alpha", "IoU_pen"),
                     ("BF@1_alpha", "BF@1"), ("BF@1.5_alpha", "BF@1.5"), ("BF@2.5_alpha", "BF@2.5")]:
        iz[dst] = iz[src]
    iz["BF_native"] = pd.read_csv(os.path.join(HERE, "results_izgc0001.csv"))["BF"]
    iz["set"] = "izgc"
    frames.append(iz)

    dpc = pd.read_csv(os.path.join(HERE, "results_dpc.csv"))
    dpc = dpc.copy()
    dpc["BF_native"] = float("nan")
    dpc["set"] = "dpc"
    frames.append(dpc)

    base = pd.read_csv(os.path.join(HERE, "results_boundary.csv"))
    base = base[~base.method.isin(["IZGC-Full", "IZGC-NoZP", "DPC-FWSN"])].copy()
    fix = os.path.join(HERE, "results_base_gapfix.csv")
    if os.path.exists(fix):
        rep = pd.read_csv(fix)
        base = base[~((base.scene.isin(rep.scene.unique())) & (base.method.isin(rep.method.unique())))]
        base = pd.concat([base, rep], ignore_index=True)
        print(f"  note: gap scenes of the point methods replaced by {os.path.basename(fix)}")
    base["BF_native"] = float("nan")
    base["set"] = "base"
    frames.append(base)

    df = pd.concat(frames, ignore_index=True)
    df = df[df.scene.isin(SCENES)]
    return df


def fmt(v, sd):
    return f"{v:.3f}±{sd:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables_by_scene.md")
    ap.add_argument("--csv", default="table_by_scene_001.csv")
    args = ap.parse_args()

    df = load()
    df.to_csv(os.path.join(HERE, args.csv), index=False)
    print(f"  wrote {args.csv}  ({len(df)} rows)")

    cols = ["BF", "BPrec", "BRec", "HD95pct", "IoU_pen", "ARI", "k_pred"]
    lines = ["# 六个合成场景的逐方法指标（0.001 步长 IZGC 口径，20 种子均值±标准差）", "",
             "边界 F1 统一用 α-shape 多边形化计算（与点标签方法同公式）；"
             "IZGC 自带等值线口径见文末附表。距离单位为场景坐标单位（场景最长边归一化为 100）。", ""]

    # ---- overall ----
    lines += ["## 总表（6 场景 × 20 种子 = 120 次）", "",
              "| 方法 | BF (α-shape 统一口径) | BF (IZGC 自带等值线) | BPrec | BRec | HD95 | IoU_pen | ARI | k_pred |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        d = df[df.method == m]
        if not len(d):
            continue
        row = f"| {m} | {fmt(d.BF.mean(), d.BF.std(ddof=1))} |"
        if d.BF_native.notna().any():
            row += f" {fmt(d.BF_native.mean(), d.BF_native.std(ddof=1))} |"
        else:
            row += " — |"
        for c in cols[1:]:
            row += f" {fmt(d[c].mean(), d[c].std(ddof=1))} |"
        lines.append(row)
    lines.append("")
    lines.append("BF (α-shape 统一口径)：所有方法（含 IZGC 两个变体）的标签均用同一 α-shape 多边形化，"
                 "再按 1.5 单位容差计算边界 F1，是跨方法唯一可比的边界指标。")
    lines.append("")
    lines.append("BF (IZGC 自带等值线)：IZGC 直接输出的等值线边界，未做多边形化；"
                 "点标签方法无此口径，记 —。该列仅用于说明 IZGC 自身两种输出形式的差异。")
    lines.append("")

    # ---- per scene ----
    for sc in SCENES:
        d0 = df[df.scene == sc]
        lines += [f"## {sc}（{TITLES[sc]}）", "",
                  "| 方法 | BF | BPrec | BRec | HD95 | IoU_pen | ARI | k_pred |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in ORDER:
            d = d0[d0.method == m]
            if not len(d):
                lines.append(f"| {m} | — | — | — | — | — | — | — |")
                continue
            row = f"| {m} |"
            for c in cols:
                row += f" {fmt(d[c].mean(), d[c].std(ddof=1))} |"
            lines.append(row)
        lines.append("")
        lines.append(f"真值簇数 k_gt = {int(d0.k_gt.iloc[0])}；n = {len(d0) // max(1, d0.method.nunique())} 种子/方法。")
        lines.append("")

    # ---- native contour (IZGC only) ----
    lines += ["## 附表：IZGC 自带等值线口径 vs 统一 α-shape 口径", "",
              "| 场景 | IZGC-Full 自带轮廓 BF | IZGC-Full 同公式 BF | 差 |",
              "|---|---|---|---|"]
    for sc in SCENES:
        d = df[(df.scene == sc) & (df.method == "IZGC-Full")]
        nat, uni = d.BF_native.mean(), d.BF.mean()
        lines.append(f"| {sc} | {nat:.3f} | {uni:.3f} | {uni - nat:+.3f} |")
    lines += ["", "自带轮廓口径偏低来自等值线栅格化后的边界带量化，"
              "跨方法比较应使用同公式口径；两种口径都应在论文中报告。", ""]

    # ---- BF tolerance sensitivity ----
    lines += ["## 附表：边界 F1 对容差的敏感性（统一口径，6 场景均值）", "",
              "| 方法 | BF@1.0 | BF@1.5 | BF@2.5 |", "|---|---|---|---|"]
    for m in ORDER:
        d = df[df.method == m]
        if not len(d):
            continue
        lines.append(f"| {m} | {d['BF@1'].mean():.3f} | {d['BF@1.5'].mean():.3f} | {d['BF@2.5'].mean():.3f} |")
    lines.append("")

    path = os.path.join(HERE, args.out)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
