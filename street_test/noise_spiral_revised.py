"""
噪声实验 - Spiral数据集
7种算法对比：IZGC, K-means++, CLIQUE, BANG, DBSCAN, GMM, DPC-FWSN
噪声水平: 0.00 ~ 0.50 (步长0.05, 共11个水平)
重复次数: 所有算法各跑30次（默认，可通过--repeats修改）

指标计算 (7个):
  - AMI:           ClusteringMetrics 自定义实现
  - ARI:           ClusteringMetrics 自定义实现
  - FMI:           ClusteringMetrics   F-Measure (加权平均)
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
    n_repeats: int = 20


ALGORITHMS = ["IZGC", "K-means++", "CLIQUE", "BANG", "DBSCAN", "GMM", "DPC-FWSN"]

# 所有算法均跑 n_repeats 次，每次使用不同 Seed

# 全样本指标（把真实噪声-1也作为一个标签参与）——主要用于补充材料/与旧结果兼容
ALL_METRIC_COLUMNS = ["AMI", "ARI", "FMI", "Macro_F1", "Macro_Jaccard", "Completeness", "V_measure"]

# 正文建议报告：只在真实非噪声点上评价聚类结构；预测为-1的干净点仍会被惩罚
INLIER_METRIC_COLUMNS = [
    "Inlier_AMI", "Inlier_ARI", "Inlier_FMI",
    "Inlier_Macro_F1", "Inlier_Macro_Jaccard",
    "Inlier_Completeness", "Inlier_V_measure",
]

# 噪声检测能力单独评价，避免把“聚类结构恢复”和“噪声识别”混为一谈
NOISE_METRIC_COLUMNS = ["Noise_Precision", "Noise_Recall", "Noise_F1"]

METRIC_COLUMNS = ALL_METRIC_COLUMNS + INLIER_METRIC_COLUMNS + NOISE_METRIC_COLUMNS


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
    Hungarian alignment for clustering labels.

    IMPORTANT:
    This function assumes y_true contains ONLY semantic cluster labels.
    Ground-truth noise points (-1) must be removed before calling it.
    Predicted -1 is retained as noise and therefore penalizes missed inliers.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    aligned = np.full_like(y_pred, -1)
    pred_nonnoise_mask = y_pred != -1
    if pred_nonnoise_mask.sum() == 0:
        return aligned

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred[pred_nonnoise_mask])

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
    """Macro-F1 and Macro-Jaccard after Hungarian alignment."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Never allow ground-truth noise to enter the semantic cluster matching.
    keep = y_true != -1
    y_true = y_true[keep]
    y_pred = y_pred[keep]

    if len(y_true) == 0:
        return np.nan, np.nan

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


def noise_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate noise identification separately.

    Positive class = noise (-1).
    At noise_ratio=0, Noise_Recall/F1 are undefined and returned as NaN.
    """
    true_noise = np.asarray(y_true) == -1
    pred_noise = np.asarray(y_pred) == -1

    tp = int(np.sum(true_noise & pred_noise))
    fp = int(np.sum(~true_noise & pred_noise))
    fn = int(np.sum(true_noise & ~pred_noise))

    precision = tp / (tp + fp) if (tp + fp) > 0 else (np.nan if true_noise.any() else 1.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(recall):
        f1 = np.nan
    elif np.isnan(precision) or (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "Noise_Precision": precision,
        "Noise_Recall": recall,
        "Noise_F1": f1,
    }


def _evaluate_core(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Seven clustering metrics for a supplied label pair."""
    cm = ClusteringMetrics.evaluate_all(y_true, y_pred)
    f1, jac = macro_f1_jaccard(y_true, y_pred)
    return {
        "AMI": float(cm["AMI"]),
        "ARI": float(cm["ARI"]),
        "FMI": float(cm["FMI"]),
        "Macro_F1": f1,
        "Macro_Jaccard": jac,
        "Completeness": float(completeness_score(y_true, y_pred)),
        "V_measure": float(v_measure_score(y_true, y_pred)),
    }


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Two-view evaluation protocol.

    (A) All-point metrics:
        Retained for compatibility with the previous experiment. Ground-truth noise
        participates as label -1 in AMI/ARI/FMI/Completeness/V-measure.

    (B) Inlier clustering metrics (recommended for the main paper):
        Only points whose ground-truth label is not -1 are used. If the algorithm
        incorrectly predicts a clean point as -1, that error remains and is penalized.

    (C) Noise detection metrics:
        Precision, recall and F1 for identifying the injected noise points.

    This separation prevents a large artificial noise class from dominating the
    interpretation of cluster-topology recovery.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ---- all-point view ----
    all_cm = ClusteringMetrics.evaluate_all(y_true, y_pred)
    # For F1/Jaccard, semantic matching explicitly excludes ground-truth noise.
    all_f1, all_jac = macro_f1_jaccard(y_true, y_pred)
    result = {
        "AMI": float(all_cm["AMI"]),
        "ARI": float(all_cm["ARI"]),
        "FMI": float(all_cm["FMI"]),
        "Macro_F1": all_f1,
        "Macro_Jaccard": all_jac,
        "Completeness": float(completeness_score(y_true, y_pred)),
        "V_measure": float(v_measure_score(y_true, y_pred)),
    }

    # ---- inlier-only view ----
    inlier = y_true != -1
    yt = y_true[inlier]
    yp = y_pred[inlier]
    in_cm = ClusteringMetrics.evaluate_all(yt, yp)
    in_f1, in_jac = macro_f1_jaccard(yt, yp)
    result.update({
        "Inlier_AMI": float(in_cm["AMI"]),
        "Inlier_ARI": float(in_cm["ARI"]),
        "Inlier_FMI": float(in_cm["FMI"]),
        "Inlier_Macro_F1": in_f1,
        "Inlier_Macro_Jaccard": in_jac,
        "Inlier_Completeness": float(completeness_score(yt, yp)),
        "Inlier_V_measure": float(v_measure_score(yt, yp)),
    })

    result.update(noise_detection_metrics(y_true, y_pred))

    # round finite values only
    return {
        k: (round(float(v), 4) if np.isfinite(v) else np.nan)
        for k, v in result.items()
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
        "Pred_Noise": int(np.sum(y_pred == -1)),
        "Pred_Noise_Ratio": float(np.mean(y_pred == -1)),
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means = {m: np.nanmean([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
                    stds = {m: np.nanstd([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
                print(f"    {alg:12s}: "
                      f"Inlier-ARI={means['Inlier_ARI']:.4f}±{stds['Inlier_ARI']:.4f}  "
                      f"Inlier-FMI={means['Inlier_FMI']:.4f}±{stds['Inlier_FMI']:.4f}  "
                      f"Noise-F1={means['Noise_F1']:.4f}±{stds['Noise_F1']:.4f}")

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
    """Aggregate mean/std/95%CI for each metric per (Algorithm, noise_ratio)."""
    agg_dict = {}
    for m in METRIC_COLUMNS:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
        agg_dict[f"{m}_count"] = (m, "count")

    agg_dict.update({
        "Pred_K_mean": ("Pred_K", "mean"),
        "Pred_K_std": ("Pred_K", "std"),
        "Pred_Noise_Ratio_mean": ("Pred_Noise_Ratio", "mean"),
        "Pred_Noise_Ratio_std": ("Pred_Noise_Ratio", "std"),
    })

    summary = (
        raw.groupby(["Algorithm", "noise_ratio"], as_index=False)
           .agg(**agg_dict)
           .sort_values(["Algorithm", "noise_ratio"])
           .reset_index(drop=True)
    )

    for m in METRIC_COLUMNS:
        summary[f"{m}_ci95"] = 1.96 * summary[f"{m}_std"] / np.sqrt(summary[f"{m}_count"].clip(lower=1))

    # structural failure diagnostics
    true_k_map = raw.groupby(["Algorithm", "noise_ratio"])["True_K"].first().to_dict()
    under = []
    over = []
    for _, r in summary.iterrows():
        alg, noise = r["Algorithm"], r["noise_ratio"]
        sub = raw[(raw["Algorithm"] == alg) & (raw["noise_ratio"] == noise)]
        ktrue = true_k_map[(alg, noise)]
        under.append(float(np.mean(sub["Pred_K"] < ktrue)))
        over.append(float(np.mean(sub["Pred_K"] > ktrue)))
    summary["UnderSeg_Rate"] = under
    summary["OverSeg_Rate"] = over

    numeric_cols = [c for c in summary.columns if c not in ("Algorithm",)]
    for c in numeric_cols:
        if c != "noise_ratio":
            summary[c] = summary[c].round(4)
    return summary


def print_console_table(summary: pd.DataFrame, config: NoiseExperimentConfig) -> None:
    """Print a formatted table to console."""
    print("\n" + "=" * 120)
    print("Noise Experiment Summary (Spiral, fixed Δg; main metric = Inlier-ARI)")
    print("=" * 120)

    noise_levels = config.noise_levels
    algs = summary["Algorithm"].unique()

    header = f"{'Algorithm':<14s}"
    for noise in noise_levels:
        header += f" | {noise:>5.0%} I-ARI"
    print(header)
    print("-" * len(header))

    for alg in algs:
        line = f"{alg:<14s}"
        for noise in noise_levels:
            row = summary[(summary["Algorithm"] == alg) & (summary["noise_ratio"] == noise)]
            if row.empty:
                line += f" | {'--':>10s}"
            else:
                ari = row["Inlier_ARI_mean"].values[0]
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
                ari = row["Inlier_ARI_mean"].values[0]
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


def _plot_metric_curves(summary, metric, ylabel, outpath, config, max_noise=None):
    """Reusable publication-style curve with 95% CI."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for alg in ALGORITHMS:
        sub = summary[summary["Algorithm"] == alg].sort_values("noise_ratio").copy()
        if max_noise is not None:
            sub = sub[sub["noise_ratio"] <= max_noise]
        if sub.empty or f"{metric}_mean" not in sub:
            continue
        x = sub["noise_ratio"].to_numpy(float)
        y = sub[f"{metric}_mean"].to_numpy(float)
        ci = sub[f"{metric}_ci95"].to_numpy(float) if f"{metric}_ci95" in sub else sub[f"{metric}_std"].to_numpy(float)
        ax.plot(x, y, marker=ALG_MARKERS.get(alg, "o"), color=ALG_COLORS.get(alg),
                linewidth=2.2, markersize=7, label=alg)
        ax.fill_between(x, y-ci, y+ci, alpha=0.10, color=ALG_COLORS.get(alg))
    ax.set_xlabel("Noise Ratio", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    levels = [x for x in config.noise_levels if max_noise is None or x <= max_noise]
    ax.set_xticks(levels)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_curves(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """
    Two families of figures:
    1) main-paper curves: inlier metrics, 0--30% contamination;
    2) supplementary curves: full 0--50% range and all metrics.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Main paper: focus on actual cluster-structure recovery.
    for metric in ["Inlier_ARI", "Inlier_FMI", "Inlier_Macro_F1", "Inlier_Macro_Jaccard"]:
        _plot_metric_curves(
            summary, metric, metric.replace("Inlier_", "").replace("_", " "),
            outdir / f"main_{metric.lower()}_0_30.png", config, max_noise=0.30
        )

    # Noise detection is a separate task.
    for metric in ["Noise_Precision", "Noise_Recall", "Noise_F1"]:
        _plot_metric_curves(
            summary, metric, metric.replace("_", " "),
            outdir / f"main_{metric.lower()}_0_30.png", config, max_noise=0.30
        )

    # Supplement: full range for every available score.
    for metric in ALL_METRIC_COLUMNS + INLIER_METRIC_COLUMNS:
        _plot_metric_curves(
            summary, metric, metric.replace("_", " "),
            outdir / f"supp_{metric.lower()}_0_50.png", config, max_noise=None
        )


def plot_cluster_count_transition(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """Predicted cluster count and under-segmentation rate; useful to explain IZGC failure mode."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Predicted K
    fig, ax = plt.subplots(figsize=(9, 6))
    for alg in ALGORITHMS:
        sub = summary[(summary["Algorithm"] == alg) & (summary["noise_ratio"] <= 0.30)].sort_values("noise_ratio")
        if sub.empty:
            continue
        ax.plot(sub["noise_ratio"], sub["Pred_K_mean"], marker=ALG_MARKERS.get(alg, "o"),
                color=ALG_COLORS.get(alg), linewidth=2, markersize=6, label=alg)
    ax.axhline(3, linestyle="--", linewidth=1.5, label="True K=3")
    ax.set_xlabel("Noise Ratio", fontsize=13)
    ax.set_ylabel("Mean Number of Predicted Clusters", fontsize=13)
    ax.set_xticks([x for x in config.noise_levels if x <= 0.30])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "main_predicted_cluster_count_0_30.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Under-segmentation rate
    fig, ax = plt.subplots(figsize=(9, 6))
    for alg in ALGORITHMS:
        sub = summary[(summary["Algorithm"] == alg) & (summary["noise_ratio"] <= 0.30)].sort_values("noise_ratio")
        if sub.empty:
            continue
        ax.plot(sub["noise_ratio"], sub["UnderSeg_Rate"], marker=ALG_MARKERS.get(alg, "o"),
                color=ALG_COLORS.get(alg), linewidth=2, markersize=6, label=alg)
    ax.set_xlabel("Noise Ratio", fontsize=13)
    ax.set_ylabel("Under-segmentation Rate", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([x for x in config.noise_levels if x <= 0.30])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "main_undersegmentation_rate_0_30.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_izgc_diagnostic(summary: pd.DataFrame, outdir: Path, config: NoiseExperimentConfig) -> None:
    """IZGC-only diagnostic: inlier quality + K transition, highlighting 10--15% transition region."""
    outdir.mkdir(parents=True, exist_ok=True)
    izgc = summary[(summary["Algorithm"] == "IZGC") & (summary["noise_ratio"] <= 0.30)].sort_values("noise_ratio")
    if izgc.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for metric, marker in [("Inlier_ARI", "o"), ("Inlier_FMI", "s"), ("Inlier_V_measure", "^")]:
        ax.plot(izgc["noise_ratio"], izgc[f"{metric}_mean"], marker=marker, linewidth=2.2,
                markersize=7, label=metric.replace("Inlier_", ""))
    ax.axvspan(0.10, 0.15, alpha=0.08, label="Transition region")
    ax.set_xlabel("Noise Ratio", fontsize=13)
    ax.set_ylabel("Inlier Clustering Score", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([x for x in config.noise_levels if x <= 0.30])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "izgc_inlier_transition_0_30.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(izgc["noise_ratio"], izgc["Pred_K_mean"], yerr=izgc["Pred_K_std"],
                marker="o", linewidth=2.2, capsize=3, label="IZGC predicted K")
    ax.axhline(3, linestyle="--", linewidth=1.5, label="True K=3")
    ax.axvspan(0.10, 0.15, alpha=0.08, label="Transition region")
    ax.set_xlabel("Noise Ratio", fontsize=13)
    ax.set_ylabel("Number of Predicted Clusters", fontsize=13)
    ax.set_xticks([x for x in config.noise_levels if x <= 0.30])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "izgc_cluster_count_transition_0_30.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_examples(config: NoiseExperimentConfig, outdir: Path) -> None:
    """Visualize the perturbation itself. Focus on the empirically important transition levels."""
    outdir.mkdir(parents=True, exist_ok=True)
    X_base, y_base = load_spiral_csv(config.spiral_csv_path)
    selected = [0.00, 0.10, 0.15, 0.25, 0.30]

    fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 4.2))
    for i, noise in enumerate(selected):
        X, y = add_noise_to_dataset(X_base, y_base, noise, config.random_seed + 1)
        ax = axes[i]
        clean = y != -1
        ax.scatter(X[clean, 0], X[clean, 1], s=8, alpha=0.75)
        if np.any(~clean):
            ax.scatter(X[~clean, 0], X[~clean, 1], s=8, alpha=0.40, c="gray", marker="x")
        ax.set_title(f"Noise={noise:.0%}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / "spiral_noise_transition_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_algorithm_for_visual(alg, X, y_true, config, seed, matlab_runner=None):
    k_true = len([v for v in np.unique(y_true) if v != -1])
    if alg == "IZGC":
        return run_izgc(X, config), matlab_runner
    if alg == "K-means++":
        return run_kmeans(X, k_true, seed), matlab_runner
    if alg == "GMM":
        return run_gmm(X, k_true, seed), matlab_runner
    if alg == "DBSCAN":
        return run_dbscan(X, config.dbscan_eps, config.dbscan_min_samples), matlab_runner
    if alg == "CLIQUE":
        return run_clique(X, config.clique_intervals, config.clique_density_threshold), matlab_runner
    if alg == "BANG":
        return run_bang(X, config.bang_levels, config.bang_density_threshold), matlab_runner
    if alg == "DPC-FWSN":
        return run_dpc_fwsn(X, config.dpc_k, k_true, matlab_runner)
    raise ValueError(alg)


def plot_izgc_transition_clusters(raw: pd.DataFrame, config: NoiseExperimentConfig, outdir: Path) -> None:
    """
    Plot IZGC cluster assignments at 0/10/15/25% noise.
    For each level choose the repetition whose Inlier_ARI is closest to the median,
    avoiding cherry-picking a best or worst run.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    X_base, y_base = load_spiral_csv(config.spiral_csv_path)
    selected = [0.00, 0.10, 0.15, 0.25]

    fig, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 4.5))
    for ax, noise in zip(axes, selected):
        sub = raw[(raw["Algorithm"] == "IZGC") & (raw["noise_ratio"] == noise)].copy()
        if sub.empty:
            ax.axis("off"); continue
        med = sub["Inlier_ARI"].median()
        chosen = sub.iloc[(sub["Inlier_ARI"] - med).abs().argsort().iloc[0]]
        seed = int(chosen["Seed"])
        X, y_true = add_noise_to_dataset(X_base, y_base, noise, seed)
        y_pred = run_izgc(X, config)

        for lab in np.unique(y_pred):
            m = y_pred == lab
            if lab == -1:
                ax.scatter(X[m, 0], X[m, 1], s=7, alpha=0.30, c="gray", marker="x")
            else:
                ax.scatter(X[m, 0], X[m, 1], s=7, alpha=0.75)
        ax.set_title(f"Noise={noise:.0%}\nARI={chosen['Inlier_ARI']:.3f}, K={int(chosen['Pred_K'])}")
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(outdir / "izgc_transition_cluster_assignments.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness on Spiral with separated inlier-clustering and noise-detection evaluation."
    )
    parser.add_argument("--outdir", type=str, default="results_noise_spiral")
    parser.add_argument("--delta-g", type=float, default=2.46)
    parser.add_argument("--repeats", type=int, default=30,
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
        plot_cluster_count_transition(summary, figdir, config)
        plot_izgc_diagnostic(summary, figdir, config)
        plot_izgc_transition_clusters(raw, config, figdir)
        print(f"  Figures: {figdir}")
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
