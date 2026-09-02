"""
多数据集7指标计算脚本
对6个数据集 (Jain, Ring, Cth, Ls, LineBlobs, 3c) 用7种算法在表格给定参数下计算7个指标。

算法: IZGC, K-means++, CLIQUE, BANG, DBSCAN, GMM, DPC-FWSN
指标: AMI, ARI, FMI, Macro_F1, Macro_Jaccard, Completeness, V_measure

数据集参数 (来自论文LaTeX表格):
  Jain:       IZGC Δg=2.826, CLIQUE l=14 dt=1, BANG L=11 dt=0.01, DBSCAN ε=8 ms=2, GMM n=1, DPC k=4
  Ring:       IZGC Δg=3.043, CLIQUE l=14 dt=1, BANG L=11 dt=0.01, DBSCAN ε=6 ms=2, GMM n=1, DPC k=20
  Cth:        IZGC Δg=2.004, CLIQUE l=17 dt=1, BANG L=11 dt=0.01, DBSCAN ε=5 ms=2, GMM n=1, DPC k=13
  Ls:         IZGC Δg=2.170, CLIQUE l=22 dt=1, BANG L=11 dt=0.01, DBSCAN ε=5 ms=2, GMM n=1, DPC k=15
  LineBlobs:  IZGC Δg=2.923, CLIQUE l=10 dt=1, BANG L=9  dt=0.01, DBSCAN ε=7 ms=2, GMM n=1, DPC k=21
  3c:         IZGC Δg=1.742, CLIQUE l=21 dt=1, BANG L=11 dt=0.01, DBSCAN ε=4 ms=2, GMM n=1, DPC k=66

用法:
  python dataset_metrics.py
  python dataset_metrics.py --no-dpc
  python dataset_metrics.py --outdir results_dataset_metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustering_metrics import ClusteringMetrics

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import completeness_score, v_measure_score

# pyclustering
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
# 配置: 数据集 & 算法参数 (来自论文LaTeX表格)
# ============================================================

ALGORITHMS = ["IZGC", "K-means++", "CLIQUE", "BANG", "DBSCAN", "GMM", "DPC-FWSN"]
METRIC_COLUMNS = ["AMI", "ARI", "FMI", "Macro_F1", "Macro_Jaccard", "Completeness", "V_measure"]

# 数据集配置: (csv文件名, 显示名称)
# 第一组: uneven density datasets
DATASETS = [
    ("jain.csv",       "Jain"),
    ("ring.csv",       "Ring"),
    ("cth.csv",        "Cth"),
    ("ls.csv",         "Ls"),
    ("lineblobs.csv",  "LineBlobs"),
    ("2d_3c.csv",      "3c"),
    # 第二组: complex-morphic datasets
    ("flame.csv",      "Flame"),
    ("spiral.csv",     "Spiral"),
    ("complex8.csv",   "Complex8"),
    ("complex9.csv",   "Complex9"),
    ("R15.csv",        "R15"),
    ("Aggregation.csv", "Aggregation"),
]

# IZGC参数: Δg (grid_size), R=30, θ=5, max_buffer_distance=0, merge_small_clusters=True
IZGC_PARAMS = {
    # 第一组
    "Jain":        {"grid_size": 2.826, "R": 30.0, "theta": 5},
    "Ring":        {"grid_size": 3.043, "R": 30.0, "theta": 5},
    "Cth":         {"grid_size": 2.004, "R": 30.0, "theta": 5},
    "Ls":          {"grid_size": 2.170, "R": 30.0, "theta": 5},
    "LineBlobs":   {"grid_size": 2.923, "R": 30.0, "theta": 5},
    "3c":          {"grid_size": 1.742, "R": 30.0, "theta": 5},
    # 第二组
    "Flame":       {"grid_size": 3.160, "R": 30.0, "theta": 5},
    "Spiral":      {"grid_size": 2.460, "R": 30.0, "theta": 5},
    "Complex8":    {"grid_size": 1.144, "R": 30.0, "theta": 5},
    "Complex9":    {"grid_size": 0.813, "R": 30.0, "theta": 5},
    "R15":         {"grid_size": 0.803, "R": 30.0, "theta": 5},
    "Aggregation": {"grid_size": 1.491, "R": 30.0, "theta": 5},
}

# CLIQUE参数: intervals, density_threshold
CLIQUE_PARAMS = {
    # 第一组
    "Jain":        {"intervals": 14, "density_threshold": 1},
    "Ring":        {"intervals": 14, "density_threshold": 1},
    "Cth":         {"intervals": 17, "density_threshold": 1},
    "Ls":          {"intervals": 22, "density_threshold": 1},
    "LineBlobs":   {"intervals": 10, "density_threshold": 1},
    "3c":          {"intervals": 21, "density_threshold": 1},
    # 第二组
    "Flame":       {"intervals": 13, "density_threshold": 1},
    "Spiral":      {"intervals": 15, "density_threshold": 1},
    "Complex8":    {"intervals": 36, "density_threshold": 1},
    "Complex9":    {"intervals": 39, "density_threshold": 1},
    "R15":         {"intervals": 24, "density_threshold": 2},
    "Aggregation": {"intervals": 20, "density_threshold": 1},
}

# BANG参数: levels, density_threshold
BANG_PARAMS = {
    # 第一组
    "Jain":        {"levels": 11, "density_threshold": 0.01},
    "Ring":        {"levels": 11, "density_threshold": 0.01},
    "Cth":         {"levels": 11, "density_threshold": 0.01},
    "Ls":          {"levels": 11, "density_threshold": 0.01},
    "LineBlobs":   {"levels": 9,  "density_threshold": 0.01},
    "3c":          {"levels": 11, "density_threshold": 0.01},
    # 第二组
    "Flame":       {"levels": 10, "density_threshold": 0.01},
    "Spiral":      {"levels": 10, "density_threshold": 0.01},
    "Complex8":    {"levels": 13, "density_threshold": 0.01},
    "Complex9":    {"levels": 13, "density_threshold": 0.01},
    "R15":         {"levels": 13, "density_threshold": 0.01},
    "Aggregation": {"levels": 11, "density_threshold": 0.01},
}

# DBSCAN参数: eps, min_samples
DBSCAN_PARAMS = {
    # 第一组
    "Jain":        {"eps": 8.0, "min_samples": 2},
    "Ring":        {"eps": 6.0, "min_samples": 2},
    "Cth":         {"eps": 5.0, "min_samples": 2},
    "Ls":          {"eps": 5.0, "min_samples": 2},
    "LineBlobs":   {"eps": 7.0, "min_samples": 2},
    "3c":          {"eps": 4.0, "min_samples": 2},
    # 第二组
    "Flame":       {"eps": 10.0, "min_samples": 9},
    "Spiral":      {"eps": 4.0,  "min_samples": 2},
    "Complex8":    {"eps": 3.0,  "min_samples": 2},
    "Complex9":    {"eps": 3.0,  "min_samples": 10},
    "R15":         {"eps": 5.0,  "min_samples": 30},
    "Aggregation": {"eps": 6.0,  "min_samples": 11},
}

# GMM参数: n_components (来自LaTeX表格的n=值), n_init=1, covariance_type=full
GMM_N_INIT = 1
GMM_N_COMPONENTS = {
    # 第一组 (n=1 表示 n_init=1, n_components=真实簇数)
    "Jain":        None,   # None = auto (= true K)
    "Ring":        None,
    "Cth":         None,
    "Ls":          None,
    "LineBlobs":   None,
    "3c":          None,
    # 第二组
    "Flame":       2,
    "Spiral":      3,
    "Complex8":    8,
    "Complex9":    28,
    "R15":         10,
    "Aggregation": 10,
}

# DPC-FWSN参数: k
DPC_K = {
    # 第一组
    "Jain":        4,
    "Ring":        20,
    "Cth":         13,
    "Ls":          15,
    "LineBlobs":   21,
    "3c":          34,
    # 第二组
    "Flame":       44,
    "Spiral":      4,
    "Complex8":    12,
    "Complex9":    14,
    "R15":         18,
    "Aggregation": 30,
}

# K-means++: n_clusters=真实簇数, n_init=20, max_iter=300
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 300

RANDOM_SEED = 42


# ============================================================
# 工具函数
# ============================================================

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    """min-max -> [0, 100]"""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    X_norm = np.zeros_like(X)
    X_norm[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 0
    X_norm[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 0
    return X_norm


def hungarian_align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    aligned = np.full(len(y_pred), -1, dtype=int)
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


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载CSV数据集 (x, y, target), 归一化到[0,100]
    Returns: X, y_raw (原始标签), y_zero (0基标签)
    """
    df = pd.read_csv(csv_path)
    X = df[["x", "y"]].values.astype(float)
    y_raw = df["target"].values.astype(int)
    # 0基标签 (供Python算法和指标计算用)
    unique_labels = np.unique(y_raw)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_zero = np.array([label_map[v] for v in y_raw])
    X = normalize_coordinates(X)
    return X, y_raw, y_zero


# ============================================================
# 算法运行函数
# ============================================================

def run_izgc(X, grid_size, R=30.0, theta=5, merge_small_clusters=True):
    from scipy.spatial import cKDTree
    import ContourBasedClusteringWithoutZ22 as CBC
    points = np.array(X)
    tree = cKDTree(points)
    density = np.array([len(tree.query_ball_point(p, R)) for p in points])
    combined = [(*pt, density[i], i) for i, pt in enumerate(points)]
    model = CBC.ContourBasedClusteringWithoutZ(
        combined, grid_size=grid_size, max_buffer_distance=0,
        min_cluster_size=theta, interactive=False,
        merge_small_clusters=merge_small_clusters,
    )
    model.perform_clustering()
    return np.asarray(model.get_predicted_labels(), dtype=int)


def run_kmeans(X, n_clusters, n_init=20, max_iter=300, random_state=42):
    return KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init,
                  max_iter=max_iter, random_state=random_state,
                  algorithm='elkan').fit_predict(X)


def run_gmm(X, n_components, covariance_type='full', n_init=1, random_state=42):
    return GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                           n_init=n_init, random_state=random_state,
                           max_iter=200).fit_predict(X)


def run_dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit_predict(X)


def run_clique(X, intervals, density_threshold):
    if not PYCLUSTERING_AVAILABLE:
        raise RuntimeError("pyclustering not available")
    inst = CLIQUEInstance(X.tolist(), intervals, density_threshold)
    inst.process()
    clusters = inst.get_clusters()
    labels = np.full(len(X), -1, dtype=int)
    for cid, pts in enumerate(clusters):
        for idx in pts:
            labels[idx] = cid
    return labels


def run_bang(X, levels, density_threshold):
    if not PYCLUSTERING_AVAILABLE:
        raise RuntimeError("pyclustering not available")
    inst = BANGInstance(X.tolist(), levels, density_threshold=density_threshold)
    inst.process()
    clusters = inst.get_clusters()
    labels = np.full(len(X), -1, dtype=int)
    for cid, pts in enumerate(clusters):
        for idx in pts:
            labels[idx] = cid
    return labels


def run_dpc_fwsn(X, y_true_raw, k, matlab_runner=None):
    """DPC-FWSN (MATLAB)
    MATLAB索引从1开始，标签中不能含0。
    将标签重映射到1-based (LabelEncoder + 1)，返回后再映射回0-based。
    """
    if not MATLAB_AVAILABLE:
        raise RuntimeError("matlab_integration not available")
    if matlab_runner is None:
        matlab_runner = MatlabRunner(method='auto')
    unique_labels = np.unique(y_true_raw)
    n_clusters = len(unique_labels)
    # 重映射: 原始标签 -> 1, 2, 3, ...
    remap = {old: new + 1 for new, old in enumerate(unique_labels)}
    y_remapped = np.array([remap[v] for v in y_true_raw], dtype=float)
    data_with_labels = np.column_stack([X, y_remapped])
    labels, metrics = matlab_runner.run_algorithm_pr(data_with_labels, n_clusters, k)
    if labels is None:
        raise RuntimeError("DPC-FWSN returned None")
    # MATLAB返回1-based标签, 转为0-based
    return np.asarray(labels, dtype=int) - 1, matlab_runner


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute 7 metrics for 7 algorithms on 6 datasets with paper-specified parameters."
    )
    parser.add_argument("--outdir", type=str, default="results_dataset_metrics")
    parser.add_argument("--no-dpc", action="store_true", help="Skip DPC-FWSN")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    algorithms = args.algorithms if args.algorithms else list(ALGORITHMS)
    if args.no_dpc and "DPC-FWSN" in algorithms:
        algorithms = [a for a in algorithms if a != "DPC-FWSN"]
        print("  DPC-FWSN skipped (--no-dpc)")

    data_dir = Path(PROJECT_ROOT) / "data"

    print("=" * 90)
    print("Dataset Metrics Computation")
    print(f"  Datasets: {[d[1] for d in DATASETS]}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Metrics: {METRIC_COLUMNS}")
    print("=" * 90)

    # 初始化MATLAB
    matlab_runner = None
    if "DPC-FWSN" in algorithms and MATLAB_AVAILABLE:
        print("  Initializing MATLAB engine for DPC-FWSN...")
        try:
            matlab_runner = MatlabRunner(method='auto')
            print("  MATLAB engine initialized.")
        except Exception as e:
            print(f"  WARNING: MATLAB init failed: {e}")
            algorithms = [a for a in algorithms if a != "DPC-FWSN"]

    all_rows = []
    # 保存每个数据集的原始数据+预测标签 (NPZ)
    dataset_predictions = {}

    for csv_name, display_name in DATASETS:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            print(f"\n  WARNING: {csv_path} not found, skipping {display_name}")
            continue

        print(f"\n{'─' * 90}")
        print(f"  Dataset: {display_name} ({csv_name})")

        X, y_raw, y_true = load_dataset(str(csv_path))
        k_true = len(np.unique(y_true))
        print(f"  N={len(X)}, K={k_true}, labels(raw)={np.unique(y_raw)}, labels(0-based)={np.unique(y_true)}")
        print(f"  X range: [{X[:,0].min():.2f}, {X[:,0].max():.2f}] x [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")

        dataset_predictions[display_name] = {
            "X": X, "y_true": y_true, "predictions": {}
        }

        # ---- IZGC ----
        if "IZGC" in algorithms:
            p = IZGC_PARAMS[display_name]
            print(f"  [IZGC] Δg={p['grid_size']}, R={p['R']}, θ={p['theta']} ...", end=" ")
            try:
                y_pred = run_izgc(X, p['grid_size'], p['R'], p['theta'])
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "IZGC", "N": len(X), "True_K": k_true,
                       "Pred_K": len([v for v in np.unique(y_pred) if v != -1]), **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["IZGC"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "IZGC", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

        # ---- K-means++ ----
        if "K-means++" in algorithms:
            print(f"  [K-means++] n_clusters={k_true}, n_init={KMEANS_N_INIT} ...", end=" ")
            try:
                y_pred = run_kmeans(X, k_true, n_init=KMEANS_N_INIT,
                                    max_iter=KMEANS_MAX_ITER, random_state=RANDOM_SEED)
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "K-means++", "N": len(X),
                       "True_K": k_true, "Pred_K": k_true, **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["K-means++"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "K-means++", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

        # ---- CLIQUE ----
        if "CLIQUE" in algorithms and PYCLUSTERING_AVAILABLE:
            p = CLIQUE_PARAMS[display_name]
            print(f"  [CLIQUE] l={p['intervals']}, dt={p['density_threshold']} ...", end=" ")
            try:
                y_pred = run_clique(X, p['intervals'], p['density_threshold'])
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "CLIQUE", "N": len(X),
                       "True_K": k_true,
                       "Pred_K": len([v for v in np.unique(y_pred) if v != -1]), **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["CLIQUE"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "CLIQUE", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

        # ---- BANG ----
        if "BANG" in algorithms and PYCLUSTERING_AVAILABLE:
            p = BANG_PARAMS[display_name]
            print(f"  [BANG] L={p['levels']}, dt={p['density_threshold']} ...", end=" ")
            try:
                y_pred = run_bang(X, p['levels'], p['density_threshold'])
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "BANG", "N": len(X),
                       "True_K": k_true,
                       "Pred_K": len([v for v in np.unique(y_pred) if v != -1]), **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["BANG"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "BANG", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

        # ---- DBSCAN ----
        if "DBSCAN" in algorithms:
            p = DBSCAN_PARAMS[display_name]
            print(f"  [DBSCAN] ε={p['eps']}, ms={p['min_samples']} ...", end=" ")
            try:
                y_pred = run_dbscan(X, p['eps'], p['min_samples'])
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "DBSCAN", "N": len(X),
                       "True_K": k_true,
                       "Pred_K": len([v for v in np.unique(y_pred) if v != -1]), **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["DBSCAN"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "DBSCAN", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

        # ---- GMM ----
        if "GMM" in algorithms:
            gmm_n = GMM_N_COMPONENTS.get(display_name)
            if gmm_n is None:
                gmm_n = k_true  # auto = true K
            print(f"  [GMM] n_components={gmm_n}, n_init={GMM_N_INIT} ...", end=" ")
            try:
                y_pred = run_gmm(X, gmm_n, n_init=GMM_N_INIT, random_state=RANDOM_SEED)
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "GMM", "N": len(X),
                       "True_K": k_true, "Pred_K": gmm_n, **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["GMM"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "GMM", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

        # ---- DPC-FWSN ----
        if "DPC-FWSN" in algorithms and matlab_runner is not None:
            k_val = DPC_K[display_name]
            print(f"  [DPC-FWSN] k={k_val} ...", end=" ")
            try:
                # run_dpc_fwsn内部已将标签重映射为1-based并返回0-based标签
                y_pred, matlab_runner = run_dpc_fwsn(X, y_raw, k_val, matlab_runner)
                m = evaluate_clustering(y_true, y_pred)
                print(f"ARI={m['ARI']:.4f} FMI={m['FMI']:.4f} F1={m['Macro_F1']:.4f}")
                row = {"Dataset": display_name, "Algorithm": "DPC-FWSN", "N": len(X),
                       "True_K": k_true,
                       "Pred_K": len([v for v in np.unique(y_pred) if v != -1]), **m}
                all_rows.append(row)
                dataset_predictions[display_name]["predictions"]["DPC-FWSN"] = y_pred
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({"Dataset": display_name, "Algorithm": "DPC-FWSN", "N": len(X),
                                 "True_K": k_true, "Pred_K": -1,
                                 **{mc: -1.0 for mc in METRIC_COLUMNS}})

    # 关闭MATLAB
    if matlab_runner is not None:
        try: matlab_runner.close()
        except: pass

    # ============================================================
    # 保存结果
    # ============================================================

    df = pd.DataFrame(all_rows)

    # 1. CSV: 全部结果
    csv_path = outdir / "dataset_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 90}")
    print(f"  CSV saved: {csv_path} ({len(df)} rows)")

    # 2. 控制台表格: 每数据集 × 算法 × 7指标
    print(f"\n{'=' * 120}")
    print("Results Summary")
    print(f"{'=' * 120}")
    for ds_name in [d[1] for d in DATASETS]:
        ds_rows = df[df["Dataset"] == ds_name]
        if ds_rows.empty:
            continue
        print(f"\n  [{ds_name}]")
        header = f"  {'Algorithm':<14s}" + "".join([f" {m:>12s}" for m in METRIC_COLUMNS])
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in ds_rows.iterrows():
            line = f"  {row['Algorithm']:<14s}"
            for m in METRIC_COLUMNS:
                val = row[m]
                if val == -1.0:
                    line += f" {'ERROR':>12s}"
                else:
                    line += f" {val:>12.4f}"
            print(line)

    # 3. LaTeX表格: 每数据集一个子表 (7指标)
    tex_lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\caption{Seven metrics comparison of different algorithms on various datasets}",
        r"\label{tab:dataset_metrics_7metrics}",
        r"\footnotesize",
        r"\begin{tabular*}{\linewidth}{@{} l" + "c" * len(METRIC_COLUMNS) + r" @{}}",
        r"\toprule",
        r"Algorithm & " + " & ".join(METRIC_COLUMNS) + r" \\",
        r"\midrule",
    ]

    for ds_name in [d[1] for d in DATASETS]:
        ds_rows = df[df["Dataset"] == ds_name]
        if ds_rows.empty:
            continue
        tex_lines.append(r"\addlinespace")
        tex_lines.append(r"\multicolumn{" + str(len(METRIC_COLUMNS) + 1) + r"}{l}{\textbf{" + ds_name + r"}} \\")
        for _, row in ds_rows.iterrows():
            cells = []
            for m in METRIC_COLUMNS:
                v = row[m]
                if v == -1.0:
                    cells.append("--")
                else:
                    # 加粗最佳值 (后续手动调整也可)
                    cells.append(f"{v:.4f}")
            tex_lines.append(f"{row['Algorithm']} & " + " & ".join(cells) + r" \\")
        tex_lines.append(r"\addlinespace")

    tex_lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    tex_path = outdir / "dataset_metrics_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"\n  LaTeX table saved: {tex_path}")

    # 4. 保存原始数据 + 预测标签 (NPZ)
    data_dir_out = outdir / "datasets"
    data_dir_out.mkdir(parents=True, exist_ok=True)
    for ds_name, rec in dataset_predictions.items():
        save_dict = {
            "X": rec["X"],
            "y_true": rec["y_true"],
        }
        for alg, y_pred in rec["predictions"].items():
            save_dict[f"pred_{alg}"] = y_pred
        np.savez_compressed(data_dir_out / f"{ds_name}.npz", **save_dict)
    print(f"  Datasets + predictions saved: {data_dir_out} ({len(dataset_predictions)} files)")

    # 5. JSON (方便程序读取)
    json_path = outdir / "dataset_metrics.json"
    result_list = []
    for _, row in df.iterrows():
        result_list.append({
            "dataset": row["Dataset"],
            "algorithm": row["Algorithm"],
            "n": int(row["N"]),
            "true_k": int(row["True_K"]),
            "pred_k": int(row["Pred_K"]),
            "metrics": {m: float(row[m]) for m in METRIC_COLUMNS},
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {json_path}")

    # 6. 保存参数配置
    config_path = outdir / "parameters.json"
    config_data = {
        "IZGC": {k: v for k, v in IZGC_PARAMS.items()},
        "CLIQUE": {k: v for k, v in CLIQUE_PARAMS.items()},
        "BANG": {k: v for k, v in BANG_PARAMS.items()},
        "DBSCAN": {k: v for k, v in DBSCAN_PARAMS.items()},
        "GMM": {"n_init": GMM_N_INIT, "n_components": "auto (=true K)"},
        "DPC-FWSN": {k: v for k, v in DPC_K.items()},
        "K-means++": {"n_init": KMEANS_N_INIT, "max_iter": KMEANS_MAX_ITER, "n_clusters": "auto (=true K)"},
        "random_seed": RANDOM_SEED,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    print(f"  Parameters saved: {config_path}")

    print(f"\n{'=' * 90}")
    print("Done!")
    print(f"  Output: {outdir}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
