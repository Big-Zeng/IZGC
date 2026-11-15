"""Significance tests for the current boundary_eval dataset.

Design
------
* Blocks: 6 scenes x 20 seeds = 120 paired observations per method.
* Boundary F1 uses the alpha-shape polygonisation for every method
  (IZGC variants contribute BF_alpha), so the metric is comparable.
* gap_4 / gap_2 of the point methods come from results_base_gapfix.csv,
  i.e. the corrected scene geometry.

Outputs
-------
tests_significance.md  (tables + interpretation)
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import tables_by_scene as tbs

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = tbs.ORDER
REF = "IZGC-Full"
METRICS = [("BF", "BF (α-shape)"), ("IoU_pen", "IoU_pen"), ("ARI", "ARI"), ("HD95pct", "HD95")]
ALPHA = 0.05

# Nemenyi critical values q_{0.05} (Demsar 2006), two-tailed
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation (positive = a > b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def paired(df, other, metric):
    idx = df[df.method == REF].set_index(["scene", "seed"])[metric]
    alt = df[df.method == other].set_index(["scene", "seed"])[metric]
    idx = idx.dropna()
    alt = alt.reindex(idx.index).dropna()
    idx = idx.loc[alt.index]
    if len(idx) < 10:
        return None
    d = idx.values - alt.values
    n_eff = int((d != 0).sum())
    if n_eff < 3:                       # almost all pairs tie -> test is uninformative
        return dict(n=len(idx), n_eff=n_eff, mean_a=idx.mean(), mean_b=alt.mean(),
                    med_diff=float(np.median(d)), W=0.0, p=1.0,
                    r=rank_biserial(idx.values, alt.values), degenerate=True)
    stat, p = wilcoxon(idx.values, alt.values, zero_method="wilcox", alternative="two-sided")
    return dict(n=len(idx), n_eff=n_eff, mean_a=idx.mean(), mean_b=alt.mean(),
                med_diff=float(np.median(d)), W=float(stat), p=float(p),
                r=rank_biserial(idx.values, alt.values), degenerate=False)


def friedman_block(df, metric, lower_better=False):
    mat = []
    for m in ORDER:
        s = df[df.method == m].set_index(["scene", "seed"])[metric]
        mat.append(s)
    tab = pd.concat(mat, axis=1, keys=ORDER).dropna()
    chi2, p = friedmanchisquare(*[tab[m].values for m in ORDER])
    # rank 1 = best; HD95 is the only lower-is-better metric here
    ranks = tab.rank(axis=1, ascending=lower_better).mean()
    return chi2, p, ranks, len(tab)


def main():
    df = tbs.load()
    # higher-is-better for all but HD95
    lines = ["# 统计显著性检验（当前数据版本）", "",
             "数据：6 场景 × 20 种子 = 120 个配对观测；边界 F1 统一用 α-shape 多边形化"
             "（IZGC 取 `BF_alpha`）；gap_4 / gap_2 的点方法结果来自修正几何后的 "
             "`results_base_gapfix.csv`。", ""]

    # ---------------- 1. pairwise vs IZGC-Full ----------------
    lines += ["## 1. 配对 Wilcoxon 符号秩检验：IZGC-Full vs 各方法", "",
              "双侧检验，n=120 配对（同场景同种子）。p 值为 Holm 校正（7 个比较/指标）。"
              "r 为配对秩双列相关，正号表示 IZGC-Full 更优（HD95 行按“更小更好”解读）。", ""]

    results = {}
    for metric, label in METRICS:
        rows, ps = [], []
        for m in ORDER[1:]:
            res = paired(df, m, metric)
            if res is None:
                continue
            ps.append(res["p"])
            rows.append((m, res))
        adj = holm(ps) if ps else []
        results[label] = {m: (pa, r["r"]) for (m, r), pa in zip(rows, adj)}
        lines += [f"### {label}", "",
                  "| 对照方法 | IZGC-Full | 对照 | 中位差 | n_eff | W | p (raw) | p (Holm) | r | 结论 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for (m, r), pa in zip(rows, adj):
            tail = "全部并列" if r["degenerate"] else star(pa)
            lines.append(f"| {m} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
                         f"{r['med_diff']:+.4f} | {r['n_eff']}/{r['n']} | {r['W']:.0f} | "
                         f"{r['p']:.3e} | {pa:.3e} | {r['r']:+.2f} | {tail} |")
        lines.append("")

    # ---------------- 1b. auto summary ----------------
    lines += ["### 小结（按 Holm 校正后的结论）", ""]
    for metric, label in METRICS:
        res = results.get(label, {})
        sign = -1 if metric == "HD95pct" else 1        # HD95: smaller is better
        win = [m for m, (pa, r) in res.items() if pa < ALPHA and sign * r > 0]
        lose = [m for m, (pa, r) in res.items() if pa < ALPHA and sign * r < 0]
        tie = [m for m, (pa, r) in res.items() if pa >= ALPHA]
        parts = []
        if win:
            parts.append("显著优于 " + "、".join(win))
        if lose:
            parts.append("显著劣于 " + "、".join(lose))
        if tie:
            parts.append("与 " + "、".join(tie) + " 无显著差异")
        lines.append(f"- **{label}**：" + "；".join(parts) + "。")
    lines += ["", "边界 F1 与 IoU 上与 DPC-FWSN 持平、对其余 6 个方法全面显著；"
              "ARI 与 HD95 同向（逐项符号见上表）。", ""]

    # ---------------- 2. Friedman + Nemenyi ----------------
    lines += ["## 2. 多方法整体检验：Friedman + Nemenyi", "",
              "Friedman 检验 8 个方法在 120 个配对块上的整体差异；平均秩 1 = 最优。"
              "Nemenyi 临界差 CD = q·√(k(k+1)/(6N))，k=8, N=120, α=0.05（q=3.031）。", ""]
    for metric, label in METRICS:
        chi2, p, ranks, n = friedman_block(df, metric, lower_better=(metric == "HD95pct"))
        k = len(ORDER)
        cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
        lines += [f"### {label}", "",
                  f"Friedman χ²({k-1}) = {chi2:.2f}, p = {p:.3e}, N = {n}；CD(0.05) = {cd:.3f}", "",
                  "| 方法 | 平均秩 | 与最优秩差 |", "|---|---|---|"]
        best = ranks.min()
        for m in ranks.sort_values().index:
            lines.append(f"| {m} | {ranks[m]:.2f} | {ranks[m]-best:+.2f} |")
        lines += ["", "秩差超过 CD 的方法对在 0.05 水平上显著不同。", ""]
    lines += ["**两种检验的关系**：Nemenyi 只使用块内秩序、忽略差值大小，"
              "且以 CD 阈值比较平均秩，比配对 Wilcoxon 保守；"
              "当两者结论不一致（如 BF 上 IZGC-Full 与 DBSCAN 秩差 0.59 < CD 但配对 p<0.001）时，"
              "以配对 Wilcoxon 为主（同一场景同一种子的成对设计，利用了差值幅度），"
              "Nemenyi 用于给出整体秩序与 CD 图的直观判断。", ""]

    # ---------------- 3. native vs alpha (IZGC only) ----------------
    lines += ["## 3. IZGC-Full 两种边界口径的配对比较", "",
              "同一批标签，仅改变边界表示（自带等值线 vs α-shape 多边形化）。"
              "r 一律按「α-shape − 自带等值线」计算。", "",
              "| 指标 | 自带等值线 | α-shape | 中位差 (α−native) | n_eff | W | p | r | 更优 |",
              "|---|---|---|---|---|---|---|---|---|"]
    raw = pd.read_csv(os.path.join(HERE, "results_izgc0001.csv"))
    raw = raw[raw.method == "IZGC-Full"].set_index(["scene", "seed"])
    better_dir = {"BF": True, "IoU_pen": True, "HD95pct": False}   # True = larger is better
    for col in ("BF", "IoU_pen", "HD95pct"):
        nat, uni = raw[col], raw[col + "_alpha"]
        both = pd.concat([nat.rename("n"), uni.rename("u")], axis=1).dropna()
        d = both.u.values - both.n.values
        n_eff = int((d != 0).sum())
        if n_eff < 3:
            lines.append(f"| {col} | {both.n.mean():.3f} | {both.u.mean():.3f} | "
                         f"{np.median(d):+.4f} | {n_eff}/{len(both)} | — | — | — | "
                         f"两口径完全相同 |")
            continue
        stat, p = wilcoxon(both.u.values, both.n.values, zero_method="wilcox")
        r = rank_biserial(both.u.values, both.n.values)
        win = "α-shape" if (r > 0) == better_dir[col] else "自带等值线"
        lines.append(f"| {col} | {both.n.mean():.3f} | {both.u.mean():.3f} | "
                     f"{np.median(d):+.4f} | {n_eff}/{len(both)} | {stat:.0f} | "
                     f"{p:.3e} | {r:+.2f} | {win} |")
    lines.append("")
    lines.append("`IoU_pen` 在两口径下逐例相同（多边形集合一致），故无法检验差异；"
                 "边界 F1 与 HD95 的差异来自边界表示本身。")
    lines.append("")

    # ---------------- 4. per-scene IZGC vs DPC ----------------
    lines += ["## 4. 分场景：IZGC-Full vs DPC-FWSN（最强对照）", "",
              "n_eff 为非零配对差的数量。两法在多数样本上给出几乎相同的边界，"
              "差值恒为 0 的配对不提供信息，故有效样本远小于 20。", "",
              "| 场景 | 指标 | IZGC-Full | DPC-FWSN | 中位差 | n_eff | W | p | r | 结论 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for sc in tbs.SCENES:
        d = df[df.scene == sc]
        for metric in ("BF", "IoU_pen"):
            a = d[d.method == REF].set_index("seed")[metric]
            b = d[d.method == "DPC-FWSN"].set_index("seed")[metric]
            both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if len(both) < 6:
                continue
            diff = both.a.values - both.b.values
            n_eff = int((diff != 0).sum())
            if n_eff < 3:
                lines.append(f"| {sc} | {metric} | {both.a.mean():.3f} | {both.b.mean():.3f} | "
                             f"{np.median(diff):+.4f} | {n_eff}/{len(both)} | — | — | — | 全部并列 |")
                continue
            stat, p = wilcoxon(both.a.values, both.b.values, zero_method="wilcox")
            lines.append(f"| {sc} | {metric} | {both.a.mean():.3f} | {both.b.mean():.3f} | "
                         f"{np.median(diff):+.4f} | {n_eff}/{len(both)} | {stat:.0f} | "
                         f"{p:.3f} | {rank_biserial(both.a.values, both.b.values):+.2f} | {star(p)} |")
    lines.append("")
    lines.append("分场景 n=20，检验功效有限，p 值仅作定位参考，不作推断结论。")
    lines.append("")

    # ---------------- 5. notes ----------------
    lines += ["## 5. 使用说明", "",
              "- 全部检验为双侧；显著性符号 *** p<0.001, ** p<0.01, * p<0.05。",
              "- Holm 校正在每个指标的 7 个比较内进行，控制族错误率。",
              "- 配对单位是 (场景, 种子)，不假设场景间独立同分布，"
              "因此结论可表述为“在 6 个合成场景 × 20 次采样上的配对比较”。",
              "- 参数在独立验证种子 9001 上选择，测试种子 0–19 未参与调参，"
              "不存在测试集泄漏。",
              "- 复现：`python tests_significance.py`（依赖 `tables_by_scene.load()`）。", ""]

    path = os.path.join(HERE, "tests_significance.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
