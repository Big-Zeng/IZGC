"""
重叠实验 - Overlapping Gaussian Blobs
仅保留 IZGC 算法，输出全部结果图表
流程:
  Step 1: 读取外部校准文件 calibration_results.json 获取IZGC最优参数
  Step 2: 使用最优参数执行两套重叠实验
    - Experiment 1: 固定σ=1.0, 改变中心距离 d=6,5,4,3,2,1
    - Experiment 2: 固定d=3, 改变簇内扩散 σ=0.5,0.75,1.0,1.25,1.5,2.0
  算法跑10次不同Seed，指标取平均，保留4位小数。

指标 (7个): AMI, ARI, FMI, Macro_F1, Macro_Jaccard, Completeness, V_measure
数据: 合成两个 Gaussian cluster, μ₁=(-d/2,0), μ₂=(d/2,0), σ²·I
      归一化到 [0,100], 每簇500点, 标签 0/1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

# 项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustering_metrics import ClusteringMetrics
from sklearn.metrics import completeness_score, v_measure_score


# ============================================================
# 0. Configuration
# ============================================================

@dataclass
class OverlapExperimentConfig:
    random_seed: int = 20260825

    # IZGC default (will be overwritten by calibration json)
    R: float = 30.0
    theta: int = 5
    delta_g_star: float = 2.0
    max_buffer_distance: int = 0

    # Experiment 1: 固定σ=1.0, 改变d
    d_values: Tuple[float, ...] = (6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    sigma_fixed: float = 1.0

    # Experiment 2: 固定d=3, 改变σ
    sigma_values: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    d_fixed: float = 3.0

    # Data
    n_per_cluster: int = 500

    # Calibration dataset: well‑separated (d=6, σ=1.0)
    calib_d: float = 6.0
    calib_sigma: float = 1.0

    # Repeats
    n_repeats: int = 1


ALGORITHMS = ["IZGC"]
METRIC_COLUMNS = ["AMI", "ARI", "FMI", "Macro_F1", "Macro_Jaccard", "Completeness", "V_measure"]


# ============================================================
# 1. Utility functions
# ============================================================

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    """min‑max -> [0, 100]"""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    X_norm = np.zeros_like(X)
    X_norm[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 0
    X_norm[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 0
    return X_norm


def hungarian_align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aligned = np.full_like(y_pred, -1)
    mask = y_pred != -1
    if mask.sum() == 0:
        return aligned
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred[mask])
    M = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            M[i, j] = np.sum((y_true == t) & (y_pred == p))
    row_ind, col_ind = linear_sum_assignment(-M)
    pred_to_true = {pred_labels[c]: true_labels[r] for r, c in zip(row_ind, col_ind)}
    next_label = int(true_labels.max()) + 1 if len(true_labels) else 0
    for p in pred_labels:
        if p in pred_to_true:
            aligned[y_pred == p] = pred_to_true[p]
        else:
            aligned[y_pred == p] = next_label
            next_label += 1
    aligned[y_pred == -1] = -1
    return aligned


def macro_f1_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    keep = y_true != -1
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if len(y_true) == 0:
        return 0.0, 0.0
    y_aligned = hungarian_align_labels(y_true, y_pred)
    classes = np.unique(y_true)
    f1s, jacs = [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_aligned == c))
        fp = np.sum((y_true != c) & (y_aligned == c))
        fn = np.sum((y_true == c) & (y_aligned != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1s.append(f1)
        jacs.append(jac)
    return float(np.mean(f1s)), float(np.mean(jacs))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = ClusteringMetrics.evaluate_all(y_true, y_pred)
    f1, jac = macro_f1_jaccard(y_true, y_pred)
    return {
        "AMI": round(float(cm["AMI"]), 4),
        "ARI": round(float(cm["ARI"]), 4),
        "FMI": round(float(cm["FMI"]), 4),
        "Macro_F1": round(f1, 4),
        "Macro_Jaccard": round(jac, 4),
        "Completeness": round(float(completeness_score(y_true, y_pred)), 4),
        "V_measure": round(float(v_measure_score(y_true, y_pred)), 4),
    }


def compute_overlap_ratio(d: float, sigma: float) -> float:
    """Bayes error 近似: 2·Φ(-d/(2σ))"""
    if sigma <= 0:
        return 0.0
    return float(2 * norm.cdf(-d / (2 * sigma)))


# ============================================================
# 2. Data generation
# ============================================================

def generate_overlapping_blobs(
    seed: int,
    d: float,
    sigma: float = 1.0,
    n_per_cluster: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """两个 Gaussian cluster, 归一化到 [0,100], 标签 0/1"""
    rng = np.random.default_rng(seed)
    mu1 = np.array([-d / 2, 0.0])
    mu2 = np.array([ d / 2, 0.0])
    cov = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
    X1 = rng.multivariate_normal(mu1, cov, n_per_cluster)
    X2 = rng.multivariate_normal(mu2, cov, n_per_cluster)
    X = np.vstack([X1, X2])
    y = np.r_[np.zeros(n_per_cluster, dtype=int),
              np.ones(n_per_cluster, dtype=int)]
    return normalize_coordinates(X), y


# ============================================================
# 3. IZGC Algorithm only
# ============================================================

def run_izgc(X: np.ndarray, grid_size: float, R: float, theta: int,
             max_buffer_distance: int = 0, merge_small_clusters: bool = True) -> np.ndarray:
    import ContourBasedClusteringWithoutZ22 as CBC
    points = np.array(X)
    tree = cKDTree(points)
    density = np.array([len(tree.query_ball_point(p, R)) for p in points])
    combined = [(*pt, density[i], i) for i, pt in enumerate(points)]
    model = CBC.ContourBasedClusteringWithoutZ(
        combined, grid_size=grid_size, max_buffer_distance=max_buffer_distance,
        min_cluster_size=theta, interactive=False, merge_small_clusters=merge_small_clusters,
    )
    model.perform_clustering()
    return np.asarray(model.get_predicted_labels(), dtype=int)


# ============================================================
# 5. Experiment runner (using calibrated params)
# ============================================================

def run_single_trial(
    algorithm: str,
    d: float,
    sigma: float,
    config: OverlapExperimentConfig,
    seed: int,
    calib: Dict[str, Dict],
    X: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    if X is None or y_true is None:
        X, y_true = generate_overlapping_blobs(seed, d, sigma, config.n_per_cluster)
    p = calib[algorithm]['params']

    if algorithm == "IZGC":
        y_pred = run_izgc(X, p['grid_size'], p['R'], p['theta'], merge_small_clusters=True)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    metrics = evaluate_clustering(y_true, y_pred)
    row = {
        "Algorithm": algorithm,
        "d": d,
        "sigma": sigma,
        "overlap_ratio": compute_overlap_ratio(d, sigma),
        "Seed": seed,
        "N": len(X),
        "True_K": len(np.unique(y_true)),
        "Pred_K": len([v for v in np.unique(y_pred) if v != -1]),
        **metrics,
    }
    return row, X, y_true, y_pred


def run_experiment_group(
    config: OverlapExperimentConfig,
    algorithms: List[str],
    calib: Dict[str, Dict],
    param_name: str,
    param_values: List[Tuple[float, float]],
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    frames = []
    data_records = {}

    for d, sigma in param_values:
        ov = compute_overlap_ratio(d, sigma)
        print(f"\n  --- {param_name}: d={d:.1f}, σ={sigma:.2f}, overlap={ov:.4f} ---")
        iter_start = time.time()

        for rep in range(1, config.n_repeats + 1):
            seed = config.random_seed + rep
            trial_key = f"d{d:.1f}_s{sigma:.2f}_seed{seed}"
            X_trial, y_trial = generate_overlapping_blobs(seed, d, sigma, config.n_per_cluster)
            data_records[trial_key] = {
                "d": d, "sigma": sigma, "seed": seed,
                "X": X_trial, "y_true": y_trial,
                "predictions": {},
            }

            for alg in algorithms:
                try:
                    row, _, _, y_pred = run_single_trial(
                        alg, d, sigma, config, seed, calib,
                        X=X_trial, y_true=y_trial,
                    )
                    row["Repeat"] = rep
                    frames.append(row)
                    data_records[trial_key]["predictions"][alg] = y_pred
                except Exception as e:
                    print(f"    {alg:12s} rep {rep}: ERROR - {e}")

        alg_rows = [r for r in frames if r["Algorithm"] == "IZGC" and r["d"] == d and r["sigma"] == sigma]
        if alg_rows:
            means = {m: np.mean([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
            stds = {m: np.std([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
            print(f"    IZGC: ARI={means['ARI']:.4f}±{stds['ARI']:.4f}  "
                  f"FMI={means['FMI']:.4f}±{stds['FMI']:.4f}  "
                  f"F1={means['Macro_F1']:.4f}±{stds['Macro_F1']:.4f}")

        print(f"  (elapsed: {time.time() - iter_start:.1f}s)")

    if outdir is not None:
        data_dir = outdir / "datasets" / param_name
        data_dir.mkdir(parents=True, exist_ok=True)
        for trial_key, rec in data_records.items():
            save_dict = {
                "X": rec["X"],
                "y_true": rec["y_true"],
                "d": np.array([rec["d"]]),
                "sigma": np.array([rec["sigma"]]),
                "seed": np.array([rec["seed"]]),
            }
            for alg, y_pred in rec["predictions"].items():
                save_dict[f"pred_{alg}"] = y_pred
            np.savez_compressed(data_dir / f"{trial_key}.npz", **save_dict)
        print(f"\n  Datasets saved: {data_dir} ({len(data_records)} files)")

        meta_rows = []
        for trial_key, rec in data_records.items():
            meta_rows.append({
                "trial_key": trial_key,
                "d": rec["d"], "sigma": rec["sigma"], "seed": rec["seed"],
                "n_points": len(rec["X"]),
                "algorithms": ",".join(rec["predictions"].keys()),
            })
        pd.DataFrame(meta_rows).to_csv(data_dir / "dataset_index.csv", index=False)

    return pd.DataFrame(frames)


# ============================================================
# 6. Summary & Tables
# ============================================================

def summarize_results(raw: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg_dict = {}
    for m in METRIC_COLUMNS:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    agg_dict["Pred_K_mean"] = ("Pred_K", "mean")
    agg_dict["overlap_ratio_mean"] = ("overlap_ratio", "mean")

    summary = (
        raw.groupby(["Algorithm", group_col], as_index=False)
           .agg(**agg_dict)
           .sort_values(["Algorithm", group_col])
           .reset_index(drop=True)
    )
    for m in METRIC_COLUMNS:
        summary[f"{m}_mean"] = summary[f"{m}_mean"].round(4)
        summary[f"{m}_std"] = summary[f"{m}_std"].round(4)
    summary["overlap_ratio_mean"] = summary["overlap_ratio_mean"].round(4)
    return summary


def print_console_table(summary: pd.DataFrame, group_col: str, title: str) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    values = sorted(summary[group_col].unique())
    header = f"{'Algorithm':<14s}"
    for v in values:
        header += f" | {v:>6.2f} ARI"
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        line = f"{row['Algorithm']:<14s}"
        for v in values:
            r = summary[summary[group_col]==v]
            if r.empty:
                line += f" | {'--':>10s}"
            else:
                line += f" | {r['ARI_mean'].values[0]:>10.4f}"
        print(line)
    print("=" * 100)


def generate_latex_table(summary, group_col, outdir, title, label, filename):
    outdir.mkdir(parents=True, exist_ok=True)
    values = sorted(summary[group_col].unique())
    col_spec = "l" + "c" * len(values)
    lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\caption{" + title + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular*}{\linewidth}{@{} " + col_spec + r" @{}}",
        r"\toprule",
        "Algorithm & " + " & ".join([f"{v:.2f}" for v in values]) + r" \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        row_line = row["Algorithm"]
        for v in values:
            r = summary[summary[group_col]==v]
            ari = r["ARI_mean"].values[0]
            std = r["ARI_std"].values[0]
            row_line += f" & {ari:.4f}$\\pm${std:.4f}"
        row_line += r" \\"
        lines.append(row_line)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    with open(outdir / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_latex_table_all_metrics(summary, group_col, outdir, title, label, filename):
    outdir.mkdir(parents=True, exist_ok=True)
    values = sorted(summary[group_col].unique())
    n_metrics = len(METRIC_COLUMNS)
    lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\caption{" + title + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular*}{\linewidth}{@{} lc" + " c" * len(values) + r" @{}}",
        r"\toprule",
        r"Algorithm & Metric & " + " & ".join([f"{v:.2f}" for v in values]) + r" \\",
        r"\midrule",
    ]
    for _, row_alg in summary.iterrows():
        alg_name = row_alg["Algorithm"]
        for i, metric in enumerate(METRIC_COLUMNS):
            if i == 0:
                lines.append(r"\multirow{" + str(n_metrics) + r"}{*}{" + alg_name + r"}")
            else:
                lines.append("")
            line = f" & {metric} "
            for v in values:
                r = summary[summary[group_col]==v]
                line += f" & {r[f'{metric}_mean'].values[0]:.4f}"
            line += r" \\"
            lines.append(line)
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    with open(outdir / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 7. Figures (Only IZGC)
# ============================================================

ALG_COLORS = {"IZGC": "#e74c3c"}
ALG_MARKERS = {"IZGC": "o"}


def plot_degradation_curves(summary, group_col, x_label, outdir, title_prefix):
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = summary.sort_values(group_col)
        ax.plot(sub[group_col].astype(float), sub[f"{metric}_mean"],
                marker=ALG_MARKERS["IZGC"], color=ALG_COLORS["IZGC"],
                linewidth=2, markersize=7, label="IZGC")
        ax.fill_between(sub[group_col].astype(float),
                        sub[f"{metric}_mean"] - sub[f"{metric}_std"],
                        sub[f"{metric}_mean"] + sub[f"{metric}_std"],
                        alpha=0.15, color=ALG_COLORS["IZGC"])
        ax.set_title(f"{title_prefix} ({metric})", fontsize=13)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f"{metric} Score", fontsize=12)
        ax.set_ylim(-0.15, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fname = f"{title_prefix.lower().replace(' ', '_')}_{metric.lower()}.png"
        fig.savefig(outdir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_dataset_examples(config, outdir, param_name, param_values):
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(param_values)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    for i, (d, sigma) in enumerate(param_values):
        X, y = generate_overlapping_blobs(config.random_seed, d, sigma, config.n_per_cluster)
        ax = axes[i] if n > 1 else axes
        for lab in np.unique(y):
            mask = y == lab
            ax.scatter(X[mask, 0], X[mask, 1], s=5, alpha=0.5)
        ov = compute_overlap_ratio(d, sigma)
        ax.set_title(f"d={d:.1f}, σ={sigma:.2f}\noverlap={ov:.3f}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.suptitle(f"Overlapping Blobs ({param_name})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"dataset_examples_{param_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_comparison(config, calib, outdir, param_name, selected_params):
    outdir.mkdir(parents=True, exist_ok=True)
    n_rows = len(selected_params)
    fig, axes = plt.subplots(n_rows, 1, figsize=(5, 4.2 * n_rows))
    for i, (d, sigma) in enumerate(selected_params):
        seed = config.random_seed + 1
        X, y_true = generate_overlapping_blobs(seed, d, sigma, config.n_per_cluster)
        ax = axes[i]
        try:
            _, _, _, y_pred = run_single_trial(
                "IZGC", d, sigma, config, seed, calib, X=X, y_true=y_true,
            )
        except Exception as e:
            y_pred = np.zeros(len(X), dtype=int)
            ax.text(0.5,0.5,f"ERROR:{e}",ha="center",va="center",transform=ax.transAxes)
        for lab in np.unique(y_pred):
            mask = y_pred == lab
            if lab == -1:
                ax.scatter(X[mask, 0], X[mask, 1], s=4, alpha=0.3, c='gray')
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=4, alpha=0.7)
        ov = compute_overlap_ratio(d, sigma)
        ax.set_title(f"IZGC, d={d:.1f}, σ={sigma:.2f}, overlap={ov:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"IZGC Clustering Results ({param_name})", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / f"izgc_clustering_{param_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overlap experiment — IZGC only on overlapping Gaussian blobs."
    )
    parser.add_argument("--outdir", type=str, default="results_overlap_izgc")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-comparison-plot", action="store_true")
    parser.add_argument("--calib", type=str, default="results_overlap/calibration_results.json",
                        help="Path to calibration_results.json from overlap_calibration.py")
    parser.add_argument("--exp", type=str, default="both", choices=["both", "d", "sigma"])
    args = parser.parse_args()

    config = OverlapExperimentConfig(n_repeats=args.repeats)
    algorithms = list(ALGORITHMS)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Overlap Experiment — IZGC Only")
    print(f"  Repeats: {config.n_repeats}")
    print(f"  Experiments: {args.exp}")
    print("=" * 70)

    calib_path = Path(args.calib)
    if not calib_path.exists():
        alt_path = Path(__file__).resolve().parent / calib_path.name
        if alt_path.exists():
            calib_path = alt_path
    if not calib_path.exists():
        print(f"\nERROR: calibration file not found: {calib_path}")
        print("Run overlap_calibration.py first.")
        sys.exit(1)

    print(f"\nLoading calibration: {calib_path}")
    with open(calib_path, "r", encoding="utf‑8") as f:
        calib_raw = json.load(f)

    calib = {"IZGC": {"params": calib_raw["IZGC"]["params"], "best_ari": calib_raw["IZGC"]["best_ari"]}}
    print(f"IZGC calibrated params: {calib['IZGC']['params']}  best ARI={calib['IZGC']['best_ari']:.4f}")

    with open(outdir / "config.json", "w", encoding="utf‑8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    all_raw_frames = []

    # ---- Experiment1: vary d ----
    if args.exp in ("both", "d"):
        print("\n" + "="*70)
        print("[Experiment 1] Varying d (σ=1.0 fixed)")
        print(f"d values: {config.d_values}")
        print("="*70)
        d_params = [(d, config.sigma_fixed) for d in config.d_values]
        if not args.no_figures:
            plot_dataset_examples(config, outdir / "figures", "vary_d", d_params)
        raw_d = run_experiment_group(config, algorithms, calib, "d", d_params, outdir)
        raw_d.to_csv(outdir / "izgc_exp_d_raw.csv", index=False)
        all_raw_frames.append(raw_d)
        summary_d = summarize_results(raw_d, "d")
        summary_d.to_csv(outdir / "izgc_exp_d_summary.csv", index=False)
        print_console_table(summary_d, "d", "IZGC Overlap: Varying d (σ=1.0)")
        if not args.no_figures:
            figdir = outdir / "figures"
            plot_degradation_curves(summary_d, "d", "Center distance d", figdir, "IZGC Overlap Varying d")
            if not args.no_comparison_plot:
                sel_d = [(6.0, config.sigma_fixed), (3.0, config.sigma_fixed), (1.0, config.sigma_fixed)]
                plot_clustering_comparison(config, calib, figdir, "vary_d", sel_d)
        generate_latex_table(summary_d, "d", outdir, "IZGC Overlap Robustness varying d", "tab:izgc_overlap_d", "izgc_overlap_d.tex")
        generate_latex_table_all_metrics(summary_d, "d", outdir, "IZGC full results varying d", "tab:izgc_overlap_d_full", "izgc_overlap_d_full.tex")

    # ---- Experiment2: vary sigma ----
    if args.exp in ("both", "sigma"):
        print("\n" + "="*70)
        print("[Experiment 2] Varying σ (d=3.0 fixed)")
        print(f"σ values: {config.sigma_values}")
        print("="*70)
        sigma_params = [(config.d_fixed, s) for s in config.sigma_values]
        if not args.no_figures:
            plot_dataset_examples(config, outdir / "figures", "vary_sigma", sigma_params)
        raw_s = run_experiment_group(config, algorithms, calib, "sigma", sigma_params, outdir)
        raw_s.to_csv(outdir / "izgc_exp_sigma_raw.csv", index=False)
        all_raw_frames.append(raw_s)
        summary_s = summarize_results(raw_s, "sigma")
        summary_s.to_csv(outdir / "izgc_exp_sigma_summary.csv", index=False)
        print_console_table(summary_s, "sigma", "IZGC Overlap: Varying σ (d=3.0)")
        if not args.no_figures:
            figdir = outdir / "figures"
            plot_degradation_curves(summary_s, "sigma", "Cluster spread σ", figdir, "IZGC Overlap Varying sigma")
            if not args.no_comparison_plot:
                sel_s = [(config.d_fixed, 0.5), (config.d_fixed, 1.0), (config.d_fixed, 2.0)]
                plot_clustering_comparison(config, calib, figdir, "vary_sigma", sel_s)
        generate_latex_table(summary_s, "sigma", outdir, "IZGC Overlap Robustness varying σ", "tab:izgc_overlap_sigma", "izgc_overlap_sigma.tex")
        generate_latex_table_all_metrics(summary_s, "sigma", outdir, "IZGC full results varying σ", "tab:izgc_overlap_sigma_full", "izgc_overlap_sigma_full.tex")

    if all_raw_frames:
        all_raw = pd.concat(all_raw_frames, ignore_index=True)
        all_raw.to_csv(outdir / "izgc_exp_all_raw.csv", index=False)

    print("\n" + "=" * 70)
    print("Done! Output dir:", outdir)
    print("=" * 70)


if __name__ == "__main__":
    main()
