"""
噪声实验 - Spiral数据集
7种算法对比：IZGC, K-means++, CLIQUE, BANG, DBSCAN, GMM, DPC-FWSN
噪声水平: 0.00 ~ 0.50 (步长0.05, 共11个水平)
重复次数: 所有算法各跑10次

指标计算 (7个):
  - AMI:           ClusteringMetrics 自定义实现
  - ARI:           ClusteringMetrics 自定义实现
  - FMI:           ClusteringMetrics MATLAB风格 F-Measure (加权平均)
  - Macro_F1:      Hungarian对齐后宏平均F1
  - Macro_Jaccard: Hungarian对齐后宏平均Jaccard
  - Completeness:  sklearn completeness_score
  - V_measure:     sklearn v_measure_score

数据来源: data/spiral.csv (312点, 3簇, 无噪声)
噪声方式: 在数据范围内添加均匀随机噪声点, 标签-1
所有数值保留小数点后4位
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

# 项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustering_metrics import ClusteringMetrics

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import completeness_score, v_measure_score

# pyclustering (CLIQUE, BANG)
try:
    from pyclustering.cluster.clique import clique as CLIQUEInstance
    from pyclustering.cluster.bang import bang as BANGInstance
    PYCLUSTERING_AVAILABLE = True
except ImportError:
    PYCLUSTERING_AVAILABLE = False
    print("警告: pyclustering未安装，CLIQUE和BANG将不可用")

# DPC-FWSN (MATLAB)
try:
    from matlab_integration import MatlabRunner
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("警告: matlab_integration未找到，DPC-FWSN将不可用")


# ============================================================
# 0. Configuration
# ============================================================

@dataclass
class NoiseExperimentConfig:
    random_seed: int = 20260824

    # IZGC parameters (与compareTest2.py一致)
    R: float = 30.0
    theta: int = 5
    delta_g_star: float = 2.46
    max_buffer_distance: int = 0

    # canonical space
    canonical_range: float = 100.0

    # noise levels: 0.00 ~ 0.50, step 0.05
    noise_levels: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(11))

    # Spiral dataset
    spiral_csv_path: str = "data/spiral.csv"

    # Algorithm parameters (from paper LaTeX table, Spiral column)
    clique_intervals: int = 15
    clique_density_threshold: int = 1
    bang_levels: int = 10
    bang_density_threshold: float = 0.01
    dbscan_eps: float = 4.0
    dbscan_min_samples: int = 2
    gmm_n_components: int = 3
    dpc_k: int = 4

    # Repeats for ALL algorithms
    n_repeats: int = 10


ALGORITHMS = ["IZGC", "K-means++", "CLIQUE", "BANG", "DBSCAN", "GMM", "DPC-FWSN"]

# 所有算法均跑 n_repeats 次，每次使用不同 Seed

METRIC_COLUMNS = ["AMI", "ARI", "FMI", "Macro_F1", "Macro_Jaccard", "Completeness", "V_measure"]


# ============================================================
# 1. Utility functions
# ============================================================

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    """与 compareTest2.py 完全相同的归一化方式: min-max -> [0, 100]"""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    X_norm = np.zeros_like(X)
    X_norm[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 0
    X_norm[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 0
    return X_norm


def hungarian_align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Align predicted cluster IDs to ground-truth labels using Hungarian matching.
    Noise label -1 is retained as -1 and excluded from matching.
    Unmatched predicted clusters receive new labels beyond max ground-truth label.
    """
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

    pred_to_true = {}
    for r, c in zip(row_ind, col_ind):
        pred_to_true[pred_labels[c]] = true_labels[r]

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
    """
    Macro-F1 and Macro-Jaccard after Hungarian alignment.
    Ground-truth classes are treated as the evaluation classes.
    Predicted noise (-1) counts as a false negative for the corresponding true class.
    """
    y_true = np.asarray(y_true)
    y_aligned = hungarian_align_labels(y_true, y_pred)

    classes = np.unique(y_true)
    f1s, jacs = [], []

    for c in classes:
        tp = np.sum((y_true == c) & (y_aligned == c))
        fp = np.sum((y_true != c) & (y_aligned == c))
        fn = np.sum((y_true == c) & (y_aligned != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        f1s.append(f1)
        jacs.append(jac)

    return float(np.mean(f1s)), float(np.mean(jacs))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算全部7个指标:
      AMI/ARI/FMI 来自 ClusteringMetrics.evaluate_all (与论文一致, FMI=MATLAB风格F-Measure)
      Macro_F1/Macro_Jaccard 来自 Hungarian对齐后的宏平均
      Completeness/V_measure 来自 sklearn
    所有值保留小数点后4位。
    """
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


# ============================================================
# 2. Dataset: load spiral.csv + add noise
# ============================================================

def load_spiral_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 data/spiral.csv，返回归一化后的 X 和 y_true。"""
    full_path = os.path.join(PROJECT_ROOT, csv_path)
    df = pd.read_csv(full_path)
    X = df[['x', 'y']].astype(float).values
    y_true = df['target'].values
    X = normalize_coordinates(X)
    return X, y_true


def add_noise_to_dataset(
    X: np.ndarray,
    y_true: np.ndarray,
    noise_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在已有数据集基础上添加均匀背景噪声。
    noise_ratio = n_noise / total_after_noise
    噪声点标签为 -1。
    """
    if noise_ratio <= 0:
        return X.copy(), y_true.copy()

    rng = np.random.default_rng(seed)
    n_clean = len(X)
    n_noise = int(round(noise_ratio * n_clean / max(1e-12, 1.0 - noise_ratio)))

    mins = X.min(axis=0) - 4
    maxs = X.max(axis=0) + 4
    noise = rng.uniform(mins, maxs, size=(n_noise, 2))

    X_noisy = np.vstack([X, noise])
    y_noisy = np.concatenate([y_true, np.full(n_noise, -1, dtype=int)])
    return X_noisy, y_noisy


# ============================================================
# 3. Algorithms
# ============================================================

def run_izgc(X: np.ndarray, config: NoiseExperimentConfig) -> np.ndarray:
    """IZGC: 与 compareTest2.py 完全一致的调用方式。"""
    import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ

    points = np.array(X)
    tree = cKDTree(points)
    density = np.array([len(tree.query_ball_point(p, config.R)) for p in points])
    combined_data = [(*point, density[i], i) for i, point in enumerate(points)]

    model = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=config.delta_g_star,
        max_buffer_distance=config.max_buffer_distance,
        min_cluster_size=config.theta,
        interactive=False,
        merge_small_clusters=False,
    )
    model.perform_clustering()
    labels = model.get_predicted_labels()
    return np.asarray(labels, dtype=int)


def run_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)


def run_gmm(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return GaussianMixture(n_components=k, n_init=10, random_state=seed).fit_predict(X)


def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def run_clique(X: np.ndarray, intervals: int, density_threshold: int) -> np.ndarray:
    """CLIQUE (pyclustering)"""
    if not PYCLUSTERING_AVAILABLE:
        raise RuntimeError("pyclustering not available")
    instance = CLIQUEInstance(X.tolist(), intervals, density_threshold)
    instance.process()
    clusters = instance.get_clusters()
    labels = np.full(len(X), -1, dtype=int)
    for cid, pts in enumerate(clusters):
        for idx in pts:
            labels[idx] = cid
    return labels


def run_bang(X: np.ndarray, levels: int, density_threshold: float) -> np.ndarray:
    """BANG (pyclustering)"""
    if not PYCLUSTERING_AVAILABLE:
        raise RuntimeError("pyclustering not available")
    instance = BANGInstance(X.tolist(), levels, density_threshold=density_threshold)
    instance.process()
    clusters = instance.get_clusters()
    labels = np.full(len(X), -1, dtype=int)
    for cid, pts in enumerate(clusters):
        for idx in pts:
            labels[idx] = cid
    return labels


def run_dpc_fwsn(
    X: np.ndarray,
    k: int,
    n_clusters: int,
    matlab_runner: Optional[object] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    """
    DPC-FWSN (MATLAB) — 无监督版本。
    与 run_dpc_unsupervised.py 一致：只传坐标 [x, y]，不传真实标签。
    """
    if not MATLAB_AVAILABLE:
        raise RuntimeError("matlab_integration not available")

    if matlab_runner is None:
        matlab_runner = MatlabRunner(method='auto')

    labels = matlab_runner.run_algorithm_pr_unsupervised(X, n_clusters, k)
    if labels is None:
        raise RuntimeError("DPC-FWSN returned None (MATLAB error)")
    return np.asarray(labels, dtype=int), matlab_runner


# ============================================================
# 4. Experiment runner
# ============================================================

def run_single_trial(
    algorithm: str,
    noise_ratio: float,
    config: NoiseExperimentConfig,
    seed: int,
    X_base: np.ndarray,
    y_base: np.ndarray,
    matlab_runner: Optional[object] = None,
) -> Tuple[Dict, Optional[object]]:
    """Run one trial for one algorithm at one noise level."""
    X, y_true = add_noise_to_dataset(X_base, y_base, noise_ratio, seed)
    k_true = len([v for v in np.unique(y_true) if v != -1])

    if algorithm == "IZGC":
        y_pred = run_izgc(X, config)
    elif algorithm == "K-means++":
        y_pred = run_kmeans(X, k=k_true, seed=seed)
    elif algorithm == "GMM":
        y_pred = run_gmm(X, k=k_true, seed=seed)
    elif algorithm == "DBSCAN":
        y_pred = run_dbscan(X, eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
    elif algorithm == "CLIQUE":
        y_pred = run_clique(X, config.clique_intervals, config.clique_density_threshold)
    elif algorithm == "BANG":
        y_pred = run_bang(X, config.bang_levels, config.bang_density_threshold)
    elif algorithm == "DPC-FWSN":
        y_pred, matlab_runner = run_dpc_fwsn(
            X, k=config.dpc_k, n_clusters=k_true, matlab_runner=matlab_runner
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    metrics = evaluate_clustering(y_true, y_pred)

    row = {
        "Algorithm": algorithm,
        "noise_ratio": noise_ratio,
        "Seed": seed,
        "N": len(X),
        "N_noise": int(np.sum(y_true == -1)),
        "True_K": k_true,
        "Pred_K": len([v for v in np.unique(y_pred) if v != -1]),
        **metrics,
    }
    return row, matlab_runner


def run_all_experiments(
    config: NoiseExperimentConfig,
    algorithms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the full noise experiment across all algorithms and noise levels."""
    if algorithms is None:
        algorithms = ALGORITHMS

    X_base, y_base = load_spiral_csv(config.spiral_csv_path)
    print(f"  Loaded {config.spiral_csv_path}: {len(X_base)} points, "
          f"{len(np.unique(y_base))} clusters")

    # 初始化MATLAB runner (如果需要DPC-FWSN)
    matlab_runner = None
    if "DPC-FWSN" in algorithms and MATLAB_AVAILABLE:
        print("  Initializing MATLAB engine for DPC-FWSN...")
        try:
            matlab_runner = MatlabRunner(method='auto')
            print("  MATLAB engine initialized.")
        except Exception as e:
            print(f"  WARNING: MATLAB init failed: {e}")
            print("  DPC-FWSN will be skipped.")
            algorithms = [a for a in algorithms if a != "DPC-FWSN"]

    frames = []
    total_start = time.time()

    for noise in config.noise_levels:
        print(f"\n  --- noise_ratio = {noise:.2f} ---")
        noise_start = time.time()

        for alg in algorithms:
            # 所有算法均跑 n_repeats 次，Seed 不同
            for rep in range(1, config.n_repeats + 1):
                seed = config.random_seed + rep
                try:
                    row, matlab_runner = run_single_trial(
                        alg, noise, config, seed, X_base, y_base, matlab_runner
                    )
                    row["Repeat"] = rep
                    frames.append(row)
                except Exception as e:
                    print(f"    {alg:12s} rep {rep}: ERROR - {e}")

            # 打印该算法的平均结果
            alg_rows = [r for r in frames if r["Algorithm"] == alg and r["noise_ratio"] == noise]
            if alg_rows:
                means = {m: np.mean([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
                stds = {m: np.std([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
                print(f"    {alg:12s}: "
                      f"AMI={means['AMI']:.4f}±{stds['AMI']:.4f}  "
                      f"ARI={means['ARI']:.4f}±{stds['ARI']:.4f}  "
                      f"FMI={means['FMI']:.4f}±{stds['FMI']:.4f}  "
                      f"F1={means['Macro_F1']:.4f}±{stds['Macro_F1']:.4f}")

        elapsed = time.time() - noise_start
        print(f"  (elapsed: {elapsed:.1f}s)")

    # 关闭MATLAB
    if matlab_runner is not None:
        try:
            matlab_runner.close()
        except:
            pass

    total_elapsed = time.time() - total_start
    print(f"\n  Total experiment time: {total_elapsed:.1f}s")

    return pd.DataFrame(frames)


# ============================================================
# 5. Summary & Tables
# ============================================================

def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate: mean ± std for each metric per (Algorithm, noise_ratio)."""
    agg_dict = {}
    for m in METRIC_COLUMNS:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    agg_dict["Pred_K_mean"] = ("Pred_K", "mean")

    summary = (
        raw.groupby(["Algorithm", "noise_ratio"], as_index=False)
           .agg(**agg_dict)
           .sort_values(["Algorithm", "noise_ratio"])
           .reset_index(drop=True)
    )
    # 所有指标保留4位小数
    for m in METRIC_COLUMNS:
        summary[f"{m}_mean"] = summary[f"{m}_mean"].round(4)
        summary[f"{m}_std"] = summary[f"{m}_std"].round(4)
    return summary


def print_console_table(summary: pd.DataFrame, config: NoiseExperimentConfig) -> None:
    """Print a formatted table to console."""
    print("\n" + "=" * 120)
    print("Noise Experiment Summary (Spiral, Δg=2.46)")
    print("=" * 120)

    noise_levels = config.noise_levels
    algs = summary["Algorithm"].unique()

    header = f"{'Algorithm':<14s}"
    for noise in noise_levels:
        header += f" | {noise:>5.0%} ARI"
    print(header)
    print("-" * len(header))

    for alg in algs:
        line = f"{alg:<14s}"
        for noise in noise_levels:
            row = summary[(summary["Algorithm"] == alg) & (summary["noise_ratio"] == noise)]
            if row.empty:
                line += f" | {'--':>10s}"
            else:
                ari = row["ARI_mean"].values[0]
                line += f" | {ari:>10.4f}"
        print(line)

    print("=" * 120)


def generate_latex_table_izgc(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """LaTeX table: IZGC only, all 7 metrics across noise levels."""
    outdir.mkdir(parents=True, exist_ok=True)
    izgc = summary[summary["Algorithm"] == "IZGC"].sort_values("noise_ratio")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{IZGC Noise Robustness on Spiral ($\Delta g=2.46$)}")
    lines.append(r"\label{tab:noise_spiral_izgc}")
    lines.append(r"\begin{tabular}{cccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Noise & AMI & ARI & FMI & F1 & Jaccard & Comp. & V-measure \\")
    lines.append(r"\midrule")

    for _, row in izgc.iterrows():
        noise = row["noise_ratio"]
        vals = [f"{row[f'{m}_mean']:.4f}" for m in METRIC_COLUMNS]
        lines.append(f"{noise:.0%} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(outdir / "noise_izgc_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_latex_table_comparison(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """LaTeX table: all algorithms, ARI across noise levels."""
    outdir.mkdir(parents=True, exist_ok=True)
    algs = summary["Algorithm"].unique()
    noise_levels = config.noise_levels

    col_spec = "l" + "c" * len(noise_levels)

    lines = []
    lines.append(r"\begin{table*}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Noise Robustness Comparison on Spiral Dataset (ARI)}")
    lines.append(r"\label{tab:noise_spiral_comparison}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{} " + col_spec + r" @{}}")
    lines.append(r"\toprule")

    header = "Algorithm"
    for noise in noise_levels:
        header += f" & {noise:.0%}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for alg in algs:
        row_line = alg
        for noise in noise_levels:
            row = summary[(summary["Algorithm"] == alg) & (summary["noise_ratio"] == noise)]
            if row.empty:
                row_line += " & --"
            else:
                ari = row["ARI_mean"].values[0]
                std = row["ARI_std"].values[0]
                row_line += f" & {ari:.4f}$\\pm${std:.4f}"
        row_line += r" \\"
        lines.append(row_line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")

    with open(outdir / "noise_comparison_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_latex_table_all_metrics(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """LaTeX table: all algorithms, all 7 metrics, at each noise level."""
    outdir.mkdir(parents=True, exist_ok=True)
    algs = summary["Algorithm"].unique()
    noise_levels = config.noise_levels
    n_metrics = len(METRIC_COLUMNS)

    lines = []
    lines.append(r"\begin{table*}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Full Noise Experiment Results on Spiral Dataset}")
    lines.append(r"\label{tab:noise_spiral_full}")
    lines.append(r"\begin{tabular*}{\linewidth}{@{} lc" + " c" * len(noise_levels) + r" @{}}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Metric & " + " & ".join([f"{n:.0%}" for n in noise_levels]) + r" \\")
    lines.append(r"\midrule")

    for alg in algs:
        for i, metric in enumerate(METRIC_COLUMNS):
            if i == 0:
                lines.append(r"\multirow{" + str(n_metrics) + r"}{*}{" + alg + r"}")
            else:
                lines.append("")

            row_line = f" & {metric} "
            for noise in noise_levels:
                row = summary[(summary["Algorithm"] == alg) & (summary["noise_ratio"] == noise)]
                if row.empty:
                    row_line += " & --"
                else:
                    val = row[f"{metric}_mean"].values[0]
                    row_line += f" & {val:.4f}"
            row_line += r" \\"
            lines.append(row_line)

        if alg != algs[-1]:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")

    with open(outdir / "noise_full_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. Figures
# ============================================================

ALG_COLORS = {
    "IZGC": "#e74c3c",
    "K-means++": "#3498db",
    "CLIQUE": "#f39c12",
    "BANG": "#9b59b6",
    "DBSCAN": "#2ecc71",
    "GMM": "#e67e22",
    "DPC-FWSN": "#1abc9c",
}
ALG_MARKERS = {
    "IZGC": "o",
    "K-means++": "s",
    "CLIQUE": "D",
    "BANG": "v",
    "DBSCAN": "^",
    "GMM": "P",
    "DPC-FWSN": "X",
}


def plot_degradation_curves(
    summary: pd.DataFrame,
    outdir: Path,
    config: NoiseExperimentConfig,
) -> None:
    """Plot degradation curves: one figure per metric, one curve per algorithm."""
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(9, 6))

        for alg in summary["Algorithm"].unique():
            sub = summary[summary["Algorithm"] == alg].sort_values("noise_ratio")
            if sub.empty:
                continue

            ax.plot(
                sub["noise_ratio"].astype(float),
                sub[f"{metric}_mean"],
                marker=ALG_MARKERS.get(alg, "o"),
                color=ALG_COLORS.get(alg, None),
                linewidth=2,
                markersize=7,
                label=alg,
            )
            ax.fill_between(
                sub["noise_ratio"].astype(float),
                sub[f"{metric}_mean"] - sub[f"{metric}_std"],
                sub[f"{metric}_mean"] + sub[f"{metric}_std"],
                alpha=0.10,
                color=ALG_COLORS.get(alg, None),
            )

        ax.set_title(f"Noise Robustness on Spiral ({metric})", fontsize=14)
        ax.set_xlabel("Noise Ratio", fontsize=13)
        ax.set_ylabel(f"{metric} Score", fontsize=13)
        ax.set_ylim(-0.15, 1.05)
        ax.set_xticks(config.noise_levels)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.tight_layout()

        fig.savefig(outdir / f"noise_{metric.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_combined_izgc(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """Combined figure: all 7 metrics for IZGC only."""
    outdir.mkdir(parents=True, exist_ok=True)
    izgc = summary[summary["Algorithm"] == "IZGC"].sort_values("noise_ratio")
    if izgc.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    metric_colors = plt.cm.tab10(np.linspace(0, 1, len(METRIC_COLUMNS)))

    for metric, color in zip(METRIC_COLUMNS, metric_colors):
        ax.plot(
            izgc["noise_ratio"].astype(float),
            izgc[f"{metric}_mean"],
            marker="o", color=color, linewidth=2, markersize=6, label=metric,
        )

    ax.set_title(
        f"IZGC Noise Robustness ($\\Delta g$={config.delta_g_star}, "
        f"R={config.R}, $\\theta$={config.theta})",
        fontsize=13,
    )
    ax.set_xlabel("Noise Ratio", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks(config.noise_levels)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "noise_izgc_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_examples(config: NoiseExperimentConfig, outdir: Path) -> None:
    """Save Spiral dataset examples at different noise levels."""
    outdir.mkdir(parents=True, exist_ok=True)
    X_base, y_base = load_spiral_csv(config.spiral_csv_path)

    selected = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
    fig, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 4.5))

    for i, noise in enumerate(selected):
        X, y = add_noise_to_dataset(X_base, y_base, noise, config.random_seed)
        ax = axes[i] if len(selected) > 1 else axes

        for lab in np.unique(y):
            mask = y == lab
            if lab == -1:
                ax.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.4, c='gray', label="noise")
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.7, label=f"C{lab}")

        ax.set_title(f"noise={noise:.0%}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Spiral Dataset with Different Noise Levels", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "spiral_noise_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for noise in config.noise_levels:
        X, y = add_noise_to_dataset(X_base, y_base, noise, config.random_seed)
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        for lab in np.unique(y):
            mask = y == lab
            if lab == -1:
                ax.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.45, c='gray', label="noise")
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.7, label=f"C{lab}")
        ax.set_title(f"Spiral (noise={noise:.0%})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"spiral_noise_{int(noise*100):02d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_clustering_comparison(config: NoiseExperimentConfig, outdir: Path) -> None:
    """Visual comparison of all 7 algorithms at selected noise levels."""
    outdir.mkdir(parents=True, exist_ok=True)

    selected_noise = [0.00, 0.20, 0.40]
    algs = ALGORITHMS
    X_base, y_base = load_spiral_csv(config.spiral_csv_path)

    matlab_runner = None
    if "DPC-FWSN" in algs and MATLAB_AVAILABLE:
        try:
            matlab_runner = MatlabRunner(method='auto')
        except:
            matlab_runner = None

    n_rows = len(selected_noise)
    n_cols = len(algs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    for i, noise in enumerate(selected_noise):
        seed = config.random_seed
        X, y_true = add_noise_to_dataset(X_base, y_base, noise, seed)
        k_true = len([v for v in np.unique(y_true) if v != -1])

        for j, alg in enumerate(algs):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            try:
                if alg == "IZGC":
                    y_pred = run_izgc(X, config)
                elif alg == "K-means++":
                    y_pred = run_kmeans(X, k=k_true, seed=seed)
                elif alg == "GMM":
                    y_pred = run_gmm(X, k=k_true, seed=seed)
                elif alg == "DBSCAN":
                    y_pred = run_dbscan(X, eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
                elif alg == "CLIQUE":
                    y_pred = run_clique(X, config.clique_intervals, config.clique_density_threshold)
                elif alg == "BANG":
                    y_pred = run_bang(X, config.bang_levels, config.bang_density_threshold)
                elif alg == "DPC-FWSN":
                    y_pred, matlab_runner = run_dpc_fwsn(
                        X, k=config.dpc_k, n_clusters=k_true, matlab_runner=matlab_runner
                    )
            except Exception:
                y_pred = np.zeros(len(X), dtype=int)
                ax.text(0.5, 0.5, "ERROR", ha='center', va='center', transform=ax.transAxes)

            for lab in np.unique(y_pred):
                mask = y_pred == lab
                if lab == -1:
                    ax.scatter(X[mask, 0], X[mask, 1], s=4, alpha=0.3, c='gray')
                else:
                    ax.scatter(X[mask, 0], X[mask, 1], s=4, alpha=0.7)

            if i == 0:
                ax.set_title(alg, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"noise={noise:.0%}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    if matlab_runner is not None:
        try:
            matlab_runner.close()
        except:
            pass

    fig.suptitle(f"Clustering Comparison on Spiral ($\\Delta g$={config.delta_g_star})",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "noise_clustering_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noise experiment on Spiral: 7 algorithms, 11 noise levels (0~0.50)."
    )
    parser.add_argument("--outdir", type=str, default="results_noise_spiral")
    parser.add_argument("--delta-g", type=float, default=2.46)
    parser.add_argument("--repeats", type=int, default=10,
                        help="Repeats for all algorithms (each uses a different Seed)")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                        help="Algorithms to run (default: all 7)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-comparison-plot", action="store_true")
    parser.add_argument("--no-dpc", action="store_true",
                        help="Skip DPC-FWSN (avoid MATLAB dependency)")
    args = parser.parse_args()

    config = NoiseExperimentConfig(
        delta_g_star=args.delta_g,
        n_repeats=args.repeats,
    )

    algorithms = args.algorithms if args.algorithms else list(ALGORITHMS)
    if args.no_dpc and "DPC-FWSN" in algorithms:
        algorithms = [a for a in algorithms if a != "DPC-FWSN"]
        print("  DPC-FWSN skipped (--no-dpc)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Noise Experiment on Spiral Dataset (data/spiral.csv)")
    print(f"  delta_g = {config.delta_g_star}")
    print(f"  R = {config.R}, theta = {config.theta}")
    print(f"  Noise levels: {config.noise_levels}")
    print(f"  Algorithms: {algorithms}")
    print(f"  All algorithms: {config.n_repeats} runs each (different Seeds)")
    print(f"  Metrics: {METRIC_COLUMNS}")
    print(f"  All values rounded to 4 decimal places")
    print("=" * 70)

    # 1. Dataset visualization
    if not args.no_figures:
        print("\n[1/5] Generating dataset visualizations...")
        plot_dataset_examples(config, outdir / "dataset_examples")

    # 2. Run experiments
    print("\n[2/5] Running noise experiments...")
    raw = run_all_experiments(config, algorithms=algorithms)
    raw.to_csv(outdir / "noise_experiment_raw.csv", index=False)
    print(f"\n  Raw results: {outdir / 'noise_experiment_raw.csv'} ({len(raw)} rows)")

    # 3. Summary
    print("\n[3/5] Computing summary...")
    summary = summarize_results(raw)
    summary.to_csv(outdir / "noise_experiment_summary.csv", index=False)
    print(f"  Summary: {outdir / 'noise_experiment_summary.csv'}")

    print_console_table(summary, config)

    # 4. Figures
    if not args.no_figures:
        print("\n[4/5] Generating figures...")
        figdir = outdir / "figures"
        plot_degradation_curves(summary, figdir, config)
        plot_combined_izgc(summary, figdir, config)
        print(f"  Figures: {figdir}")

        if not args.no_comparison_plot:
            print("  Generating clustering comparison plot...")
            plot_clustering_comparison(config, figdir)
    else:
        print("\n[4/5] Skipping figures.")

    # 5. LaTeX tables
    print("\n[5/5] Generating LaTeX tables...")
    generate_latex_table_izgc(summary, outdir, config)
    generate_latex_table_comparison(summary, outdir, config)
    generate_latex_table_all_metrics(summary, outdir, config)
    print(f"  Tables: {outdir}")

    print("\n" + "=" * 70)
    print("Done!")
    print(f"  Output: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
