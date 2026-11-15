"""Generate the LaTeX tables for Supplementary Section S11.

Three tables only:
    S11.1  per-scene metrics of the eight methods
    S11.2  (a) native contours vs the shared alpha-shape, (b) tolerance sensitivity
    S11.3  Friedman ranks, Nemenyi CD and Holm-corrected paired Wilcoxon

Reads table_by_scene_001.csv (already merged: IZGC on the alpha-shape
representation, point methods on the corrected gap geometry) and the same
statistics used in tests_significance.py, then writes ready-to-paste tabular
blocks.  No number is typed by hand.

Usage:
    python gen_supp_tex.py
"""
import os

import numpy as np
import pandas as pd

import tests_significance as ts
import tables_by_scene as tbs

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = tbs.SCENES
ORDER = tbs.ORDER
OUT = os.path.join(HERE, "supp_tables_s11.tex")

SHORT = {"nonconvex_c": r"nonconvex\_c", "nonconvex_u": r"nonconvex\_u",
         "gap_4": r"gap\_4", "gap_2": r"gap\_2",
         "density_3": r"density\_3", "density_8": r"density\_8"}

BODY = ["\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{0.95}"]


def ms(v, s):
    return f"{v:.3f}$\\pm${s:.3f}"


def bold_if(text, flag):
    return f"\\textbf{{{text}}}" if flag else text


def table_s11(df):
    """Per-scene metrics, 8 methods x 6 scenes."""
    metrics = [("BF", True), ("HD95pct", False), ("IoU_pen", True), ("ARI", True)]
    head = ("Method & BF$_{\\alpha}$ & HD95$_{\\alpha}$ "
            "& IoU$_{\\mathrm{pen}}$ & ARI & $\\bar{K}_{\\mathrm{pred}}$ \\\\")
    L = ["% ===== Table S11.1: per-scene metrics =====",
         "\\begin{table}[tbp]", "\\centering",
         "\\caption{Per-scene boundary and clustering metrics on the six controlled "
         "geometric scenes (mean$\\pm$standard deviation over 20 test seeds).}",
         "\\label{tab:S11_scene_metrics}"] + BODY + [
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}",
         "\\toprule", head, "\\midrule"]
    for si, sc in enumerate(SCENES):
        d = df[df.scene == sc]
        means = {m: {k: d[d.method == m][k].mean() for k, _ in metrics}
                 for m in ORDER}
        # best value per metric, compared after rounding to the printed precision
        # so that tied rows are bolded consistently
        best = {k: round((max if hi else min)(means[m][k]
                                              for m in ORDER if len(d[d.method == m])), 3)
                for k, hi in metrics}
        if si:
            L.append("\\midrule")
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{SHORT[sc]}}} "
                 f"($K_{{\\mathrm{{gt}}}}={int(d.k_gt.iloc[0])}$, 20 seeds)}} \\\\")
        for m in ORDER:
            dm = d[d.method == m]
            if not len(dm):
                continue
            row = []
            for k, _ in metrics:
                txt = ms(dm[k].mean(), dm[k].std(ddof=1))
                row.append(bold_if(txt, round(dm[k].mean(), 3) == best[k]))
            kp = dm.k_pred.mean()
            kps = dm.k_pred.std(ddof=1)
            name = f"\\textbf{{{m}}}" if m == "IZGC-Full" else m
            L.append(f"{name} & " + " & ".join(row) + f" & {kp:.2f}$\\pm${kps:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "",
          "\\vspace{2pt}",
          "\\noindent\\footnotesize{Notes: BF$_{\\alpha}$ and HD95$_{\\alpha}$ are "
          "computed after a common alpha-shape polygonisation of the cluster labels "
          "($\\alpha=0.05$, matching tolerance $\\tau=1.5$ spatial units). "
          "HD95$_{\\alpha}$ is the pooled bidirectional 95th-percentile boundary "
          "distance, expressed as a percentage of the scene diagonal. "
          "Bold marks the best value of each metric within a scene; ties are bolded "
          "jointly. The predicted cluster number is given as mean$\\pm$standard "
          "deviation over the 20 seeds ($K_{\\mathrm{gt}}=2$ throughout).}",
          "\\end{table}", ""]
    return L


def table_s12(df):
    """(a) native vs alpha, (b) tolerance sensitivity, in one float."""
    f = df[df.method == "IZGC-Full"]
    raw = pd.read_csv(os.path.join(HERE, "results_izgc0001.csv"))
    natdf = raw[raw.method == "IZGC-Full"].set_index(["scene", "seed"])
    nat = natdf["HD95pct"]
    nprec, nrec = natdf["BPrec"].groupby("scene"), natdf["BRec"].groupby("scene")

    L = ["% ===== Table S11.2: representation + tolerance =====",
         "\\begin{table}[tbp]", "\\centering",
         "\\caption{Boundary representation and evaluation sensitivity of IZGC-Full. "
         "(a) Native Marching Squares contours versus the shared alpha-shape "
         "representation of the same labels. (b) Boundary F1 under three matching "
         "tolerances for all eight methods.}",
         "\\label{tab:S11_native_alpha}"] + BODY + [
         "\\noindent\\textit{(a) Native contours versus alpha-shape}",
         "\\par\\vspace{2pt}",
         "\\begin{tabular*}{0.92\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
         "\\toprule",
         "Scene & BF$_{\\mathrm{nat}}$ & BF$_{\\alpha}$ & $P_{\\mathrm{nat}}$ "
         "& $R_{\\mathrm{nat}}$ & HD95$_{\\mathrm{nat}}$ & HD95$_{\\alpha}$ \\\\",
         "\\midrule"]
    for sc in SCENES:
        d = f[f.scene == sc]
        nh = nat.loc[sc].mean()
        L.append(f"{SHORT[sc]} & {d.BF_native.mean():.3f} & {d.BF_alpha.mean():.3f} & "
                 f"{nprec.get_group(sc).mean():.3f} & {nrec.get_group(sc).mean():.3f} & "
                 f"{nh:.3f} & {d.HD95pct_alpha.mean():.3f} \\\\")
    L += ["\\midrule",
          f"All & {f.BF_native.mean():.3f} & {f.BF_alpha.mean():.3f} & "
          f"{natdf.BPrec.mean():.3f} & {natdf.BRec.mean():.3f} & "
          f"{nat.mean():.3f} & {f.HD95pct_alpha.mean():.3f} \\\\",
          "\\bottomrule", "\\end{tabular*}", "",
          "\\vspace{6pt}",
          "\\noindent\\textit{(b) Sensitivity of boundary F1 to the matching tolerance}",
          "\\par\\vspace{2pt}",
          "\\begin{tabular*}{0.66\\textwidth}{@{\\extracolsep{\\fill}}lccc@{}}",
          "\\toprule",
          "Method & BF@1.0 & BF@1.5 & BF@2.5 \\\\", "\\midrule"]
    for m in ORDER:
        d = df[df.method == m]
        L.append(f"{m} & {d['BF@1'].mean():.3f} & {d['BF@1.5'].mean():.3f} & "
                 f"{d['BF@2.5'].mean():.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "",
          "\\vspace{2pt}",
          "\\noindent\\footnotesize{Notes: In panel (a), BF$_{\\mathrm{nat}}$ uses the "
          "accepted density contours directly and BF$_{\\alpha}$ re-polygonises the "
          "same predicted labels with the shared alpha-shape; $P_{\\mathrm{nat}}$ and "
          "$R_{\\mathrm{nat}}$ are the boundary precision and recall of the native "
          "contours. $\\mathrm{IoU}_{\\mathrm{pen}}$ is not repeated here because it "
          "is computed from the rasterised labels and is therefore identical under "
          "both representations. HD95 is the pooled bidirectional 95th-percentile "
          "boundary distance, as a percentage of the scene diagonal. In panel (b), "
          "$\\tau=1.5$ is used throughout the paper and corresponds to 1.5\\% of the "
          "100-unit scene side (about 1\\% of the diagonal).}",
          "\\end{table}", ""]
    return L


def table_s14(df):
    """Friedman ranks + Holm-corrected pairwise p-values."""
    L = ["% ===== Table S11.3: significance =====",
         "\\begin{table}[tbp]", "\\centering",
         "\\caption{Friedman test, Nemenyi critical difference ($k=8$, $N=120$), and "
         "Holm-corrected paired Wilcoxon tests against IZGC-Full.}",
         "\\label{tab:S11_significance}"] + BODY + [
         "\\noindent\\textit{(a) Mean Friedman ranks (1 = best)}",
         "\\par\\vspace{2pt}",
         "\\begin{tabular*}{0.82\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}",
         "\\toprule",
         "Method & BF$_{\\alpha}$ & IoU$_{\\mathrm{pen}}$ & ARI & HD95$_{\\alpha}$ \\\\",
         "\\midrule"]
    ranks, tests = {}, {}
    for metric in ("BF", "IoU_pen", "ARI", "HD95pct"):
        chi2, p, rk, n = ts.friedman_block(df, metric,
                                           lower_better=(metric == "HD95pct"))
        ranks[metric] = (chi2, p, rk, n)
        rows, ps = [], []
        for m in ORDER[1:]:
            res = ts.paired(df, m, metric)
            if res is None:
                continue
            rows.append((m, res))
            ps.append(res["p"])
        adj = ts.holm(ps)
        tests[metric] = {m: (pa, r) for (m, r), pa in zip(rows, adj)}
    for m in ORDER:
        cells = [f"{ranks[k2][2][m]:.2f}" for k2 in ("BF", "IoU_pen", "ARI", "HD95pct")]
        name = f"\\textbf{{{m}}}" if m == "IZGC-Full" else m
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule",
          "Friedman $\\chi^2(7)$ & "
          + " & ".join(f"{ranks[k2][0]:.2f}" for k2 in ("BF", "IoU_pen", "ARI", "HD95pct"))
          + " \\\\",
          "$p$ & " + " & ".join(
              ("$<$0.001" if ranks[k2][1] < 1e-3 else f"{ranks[k2][1]:.3f}")
              for k2 in ("BF", "IoU_pen", "ARI", "HD95pct")) + " \\\\",
          "Nemenyi CD$_{0.05}$ & \\multicolumn{4}{c}{0.958} \\\\",
          "\\bottomrule", "\\end{tabular*}", "",
          "\\vspace{6pt}",
          "\\noindent\\textit{(b) Holm-corrected paired Wilcoxon $p$ values against IZGC-Full}",
          "\\par\\vspace{2pt}",
          "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}",
          "\\toprule",
          "Metric & NoZP & DPC-FWSN & K-means++ & GMM & DBSCAN & CLIQUE & BANG \\\\",
          "\\midrule"]
    for metric, label in (("BF", "BF$_{\\alpha}$"),
                          ("IoU_pen", "IoU$_{\\mathrm{pen}}$"),
                          ("ARI", "ARI"),
                          ("HD95pct", "HD95$_{\\alpha}$")):
        cells = []
        for m in ORDER[1:]:
            pa, r = tests[metric][m]
            cells.append("$<$0.001" if pa < 1e-3 else f"{pa:.3f}")
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}", "",
          "\\vspace{2pt}",
          "\\noindent\\footnotesize{Notes: Ranks are computed within each of the 120 "
          "scene--seed blocks, so $N=120$ refers to six fixed scene geometries "
          "evaluated with 20 seeds each; the inference therefore applies to these "
          "controlled scenes and not to arbitrary spatial geometries. HD95 is "
          "inverted before ranking because lower is better. Holm correction is "
          "applied separately within each metric across the seven comparisons, "
          "and the $p$ values reported above are the Holm-adjusted ones. "
          "Nemenyi provides a family-wise view over all eight methods and is more "
          "conservative than the paired tests, which exploit the pairing and are "
          "reported as the primary evidence.}",
          "\\end{table}", ""]
    return L


def main():
    df = tbs.load()
    lines = []
    lines += table_s11(df)
    lines += table_s12(df)
    lines += table_s14(df)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT, len(lines), "lines")


if __name__ == "__main__":
    main()
