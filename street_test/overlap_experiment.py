"""
重叠实验 - Overlapping Gaussian Blobs
7种算法对比：IZGC, K-means++, CLIQUE, BANG, DBSCAN, GMM, DPC-FWSN

流程:
  Step 1: 在初始分离数据集 (d=6, σ=1.0) 上对每种算法进行参数网格搜索，
          以ARI为优化目标，求得各算法最优参数。
  Step 2: 用最优参数进行重叠实验:
    - Experiment 1: 固定σ=1.0, 改变中心距离 d=6,5,4,3,2,1
    - Experiment 2: 固定d=3, 改变簇内扩散 σ=0.5,0.75,1.0,1.25,1.5,2.0
  所有算法各跑10次 (不同Seed), 指标取平均, 保留4位小数。

指标 (7个): AMI, ARI, FMI, Macro_F1, Macro_Jaccard, Completeness, V_measure
  AMI/ARI/FMI: ClusteringMetrics (FMI=MATLAB风格F-Measure)
  Macro_F1/Macro_Jaccard: Hungarian对齐后宏平均
  Completeness/V_measure: sklearn

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
from itertools import product
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

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import completeness_score, v_measure_score, silhouette_score

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
# 0. Configuration
# ============================================================

@dataclass
class OverlapExperimentConfig:
    random_seed: int = 20260825

    # IZGC default (will be overwritten by calibration)
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

    # Calibration dataset: well-separated (d=6, σ=1.0)
    calib_d: float = 6.0
    calib_sigma: float = 1.0

    # Repeats
    n_repeats: int = 10


ALGORITHMS = ["IZGC", "K-means++", "CLIQUE", "BANG", "DBSCAN", "GMM", "DPC-FWSN"]

METRIC_COLUMNS = ["AMI", "ARI", "FMI", "Macro_F1", "Macro_Jaccard", "Completeness", "V_measure"]


# ============================================================
# 1. Utility functions
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


def generate_param_values(start, end, step):
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return [round(v, 4) for v in np.arange(start, end + step / 2, step)]


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
# 3. Algorithms
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


def run_kmeans(X, n_clusters, n_init=20, max_iter=300, random_state=42):
    return KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init,
                  max_iter=max_iter, random_state=random_state, algorithm='elkan').fit_predict(X)


def run_gmm(X, n_components, covariance_type='full', n_init=10, random_state=42):
    return GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                           n_init=n_init, random_state=random_state, max_iter=200).fit_predict(X)


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


def run_dpc_fwsn(X, k, n_clusters, matlab_runner=None):
    if not MATLAB_AVAILABLE:
        raise RuntimeError("matlab_integration not available")
    if matlab_runner is None:
        matlab_runner = MatlabRunner(method='auto')
    labels = matlab_runner.run_algorithm_pr_unsupervised(X, n_clusters, k)
    if labels is None:
        raise RuntimeError("DPC-FWSN returned None")
    return np.asarray(labels, dtype=int), matlab_runner


# ============================================================
# 4. Parameter calibration on well-separated data
# ============================================================

def calibrate_all_algorithms(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    config: OverlapExperimentConfig,
) -> Dict[str, Dict]:
    """
    Grid-search optimal parameters for each algorithm on the calibration set.
    Optimization target: ARI (ClusteringMetrics).
    Returns dict: { algorithm_name: {best_params: {...}, best_ari: float} }
    """
    print("\n" + "=" * 70)
    print("Step 1: Parameter Calibration on Well-Separated Data")
    print(f"  (d={config.calib_d}, σ={config.calib_sigma}, "
          f"overlap={compute_overlap_ratio(config.calib_d, config.calib_sigma):.4f})")
    print("=" * 70)

    k_true = len(np.unique(y_calib))
    best_params = {}

    # ---- IZGC ----
    print("\n  [IZGC] Calibrating grid_size & R & theta...")
    izgc_ranges = {
        'grid_size': generate_param_values(0.5, 5.0, 0.1),
        'R': [20, 25, 30, 35, 40],
        'theta': [3, 5, 7, 10],
    }
    best_ari_izgc = -np.inf
    best_izgc_params = None
    total_izgc = len(izgc_ranges['grid_size']) * len(izgc_ranges['R']) * len(izgc_ranges['theta'])
    count = 0
    for gs in izgc_ranges['grid_size']:
        for R in izgc_ranges['R']:
            for th in izgc_ranges['theta']:
                count += 1
                try:
                    labels = run_izgc(X_calib, gs, R, th, merge_small_clusters=True)
                    n_cls = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_cls < 2:
                        continue
                    ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
                    if ari > best_ari_izgc:
                        best_ari_izgc = ari
                        best_izgc_params = {'grid_size': gs, 'R': R, 'theta': th}
                except:
                    pass
        if count % 20 == 0:
            print(f"    progress: {count}/{total_izgc}, best ARI={best_ari_izgc:.4f}")
    best_params["IZGC"] = {'params': best_izgc_params, 'best_ari': best_ari_izgc}
    print(f"    IZGC best: {best_izgc_params} ARI={best_ari_izgc:.4f}")

    # ---- K-means++ ----
    print("\n  [K-means++] Calibrating n_init & max_iter...")
    km_ranges = {
        'n_clusters': [k_true],
        'n_init': [10, 20, 30, 50],
        'max_iter': [300, 500, 1000],
    }
    best_ari_km = -np.inf
    best_km_params = None
    for nc, ni, mi in product(km_ranges['n_clusters'], km_ranges['n_init'], km_ranges['max_iter']):
        try:
            labels = run_kmeans(X_calib, nc, ni, mi, random_state=config.random_seed)
            ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
            if ari > best_ari_km:
                best_ari_km = ari
                best_km_params = {'n_clusters': nc, 'n_init': ni, 'max_iter': mi}
        except:
            pass
    best_params["K-means++"] = {'params': best_km_params, 'best_ari': best_ari_km}
    print(f"    K-means++ best: {best_km_params} ARI={best_ari_km:.4f}")

    # ---- GMM ----
    print("\n  [GMM] Calibrating covariance_type & n_init...")
    gmm_cov_types = ['full', 'tied', 'diag', 'spherical']
    best_ari_gmm = -np.inf
    best_gmm_params = None
    for nc in [k_true]:
        for cov in gmm_cov_types:
            for ni in [5, 10, 20]:
                try:
                    labels = run_gmm(X_calib, nc, cov, ni, random_state=config.random_seed)
                    ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
                    if ari > best_ari_gmm:
                        best_ari_gmm = ari
                        best_gmm_params = {'n_components': nc, 'covariance_type': cov, 'n_init': ni}
                except:
                    pass
    best_params["GMM"] = {'params': best_gmm_params, 'best_ari': best_ari_gmm}
    print(f"    GMM best: {best_gmm_params} ARI={best_ari_gmm:.4f}")

    # ---- DBSCAN ----
    print("\n  [DBSCAN] Calibrating eps & min_samples...")
    dbscan_ranges = {
        'eps': generate_param_values(1.0, 50.0, 1.0),
        'min_samples': list(range(2, 21)),
    }
    best_ari_db = -np.inf
    best_db_params = None
    total_db = len(dbscan_ranges['eps']) * len(dbscan_ranges['min_samples'])
    count = 0
    for eps in dbscan_ranges['eps']:
        for ms in dbscan_ranges['min_samples']:
            count += 1
            try:
                labels = run_dbscan(X_calib, eps, ms)
                n_cls = len(set(labels)) - (1 if -1 in labels else 0)
                if n_cls < 2:
                    continue
                ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
                if ari > best_ari_db:
                    best_ari_db = ari
                    best_db_params = {'eps': eps, 'min_samples': ms}
            except:
                pass
        if count % 50 == 0:
            print(f"    progress: {count}/{total_db}, best ARI={best_ari_db:.4f}")
    best_params["DBSCAN"] = {'params': best_db_params, 'best_ari': best_ari_db}
    print(f"    DBSCAN best: {best_db_params} ARI={best_ari_db:.4f}")

    # ---- CLIQUE ----
    if PYCLUSTERING_AVAILABLE:
        print("\n  [CLIQUE] Calibrating intervals & density_threshold...")
        clique_ranges = {
            'intervals': list(range(4, 51)),
            'density_threshold': list(range(1, 31)),
        }
        best_ari_cl = -np.inf
        best_cl_params = None
        total_cl = len(clique_ranges['intervals']) * len(clique_ranges['density_threshold'])
        count = 0
        for iv in clique_ranges['intervals']:
            for dt in clique_ranges['density_threshold']:
                count += 1
                try:
                    labels = run_clique(X_calib, iv, dt)
                    n_cls = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_cls < 2:
                        continue
                    ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
                    if ari > best_ari_cl:
                        best_ari_cl = ari
                        best_cl_params = {'intervals': iv, 'density_threshold': dt}
                except:
                    pass
            if count % 50 == 0:
                print(f"    progress: {count}/{total_cl}, best ARI={best_ari_cl:.4f}")
        best_params["CLIQUE"] = {'params': best_cl_params, 'best_ari': best_ari_cl}
        print(f"    CLIQUE best: {best_cl_params} ARI={best_ari_cl:.4f}")

    # ---- BANG ----
    if PYCLUSTERING_AVAILABLE:
        print("\n  [BANG] Calibrating levels & density_threshold...")
        bang_ranges = {
            'levels': list(range(4, 31)),
            'density_threshold': [round(v, 2) for v in np.arange(0.01, 1.01, 0.05)],
        }
        best_ari_bg = -np.inf
        best_bg_params = None
        total_bg = len(bang_ranges['levels']) * len(bang_ranges['density_threshold'])
        count = 0
        for lv in bang_ranges['levels']:
            for dt in bang_ranges['density_threshold']:
                count += 1
                try:
                    labels = run_bang(X_calib, lv, dt)
                    n_cls = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_cls < 2:
                        continue
                    ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
                    if ari > best_ari_bg:
                        best_ari_bg = ari
                        best_bg_params = {'levels': lv, 'density_threshold': dt}
                except:
                    pass
            if count % 30 == 0:
                print(f"    progress: {count}/{total_bg}, best ARI={best_ari_bg:.4f}")
        best_params["BANG"] = {'params': best_bg_params, 'best_ari': best_ari_bg}
        print(f"    BANG best: {best_bg_params} ARI={best_ari_bg:.4f}")

    # ---- DPC-FWSN ----
    if MATLAB_AVAILABLE:
        print("\n  [DPC-FWSN] Calibrating k...")
        matlab_runner = None
        try:
            matlab_runner = MatlabRunner(method='auto')
            best_ari_dpc = -np.inf
            best_dpc_params = None
            for k in range(1, 51):
                try:
                    labels, matlab_runner = run_dpc_fwsn(X_calib, k, k_true, matlab_runner)
                    ari = float(ClusteringMetrics.evaluate_all(y_calib, labels)["ARI"])
                    print(f"    k={k}: ARI={ari:.4f}")
                    if ari > best_ari_dpc:
                        best_ari_dpc = ari
                        best_dpc_params = {'k': k}
                except Exception as e:
                    print(f"    k={k}: ERROR - {e}")
            best_params["DPC-FWSN"] = {'params': best_dpc_params, 'best_ari': best_ari_dpc}
            print(f"    DPC-FWSN best: {best_dpc_params} ARI={best_ari_dpc:.4f}")
        except Exception as e:
            print(f"    DPC-FWSN calibration failed: {e}")
        finally:
            if matlab_runner:
                try:
                    matlab_runner.close()
                except:
                    pass

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Calibration Summary")
    print("=" * 70)
    print(f"{'Algorithm':<14s} {'Best ARI':<10s} {'Best Params'}")
    print("-" * 70)
    for alg in ALGORITHMS:
        if alg in best_params and best_params[alg]['params']:
            print(f"{alg:<14s} {best_params[alg]['best_ari']:<10.4f} {best_params[alg]['params']}")
        else:
            print(f"{alg:<14s} {'N/A':<10s} (not available)")
    print("=" * 70)

    return best_params


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
    matlab_runner: Optional[object] = None,
    X: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> Tuple[Dict, Optional[object], np.ndarray, np.ndarray, np.ndarray]:
    if X is None or y_true is None:
        X, y_true = generate_overlapping_blobs(seed, d, sigma, config.n_per_cluster)
    k_true = len(np.unique(y_true))
    p = calib[algorithm]['params']

    if algorithm == "IZGC":
        y_pred = run_izgc(X, p['grid_size'], p['R'], p['theta'], merge_small_clusters=True)
    elif algorithm == "K-means++":
        y_pred = run_kmeans(X, p['n_clusters'], p['n_init'], p['max_iter'], random_state=seed)
    elif algorithm == "GMM":
        y_pred = run_gmm(X, p['n_components'], p['covariance_type'], p['n_init'], random_state=seed)
    elif algorithm == "DBSCAN":
        y_pred = run_dbscan(X, p['eps'], p['min_samples'])
    elif algorithm == "CLIQUE":
        y_pred = run_clique(X, p['intervals'], p['density_threshold'])
    elif algorithm == "BANG":
        y_pred = run_bang(X, p['levels'], p['density_threshold'])
    elif algorithm == "DPC-FWSN":
        y_pred, matlab_runner = run_dpc_fwsn(X, p['k'], k_true, matlab_runner)
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
        "True_K": k_true,
        "Pred_K": len([v for v in np.unique(y_pred) if v != -1]),
        **metrics,
    }
    return row, matlab_runner, X, y_true, y_pred


def run_experiment_group(
    config: OverlapExperimentConfig,
    algorithms: List[str],
    calib: Dict[str, Dict],
    param_name: str,
    param_values: List[Tuple[float, float]],
    matlab_runner: Optional[object] = None,
    outdir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[object]]:
    frames = []
    # 保存每个 (d, sigma, seed) 的原始数据 + 每算法预测标签
    # data_records: { "d{d}_s{sigma}_seed{seed}": {"X": ..., "y_true": ..., "predictions": {alg: y_pred}} }
    data_records = {}

    for d, sigma in param_values:
        ov = compute_overlap_ratio(d, sigma)
        print(f"\n  --- {param_name}: d={d:.1f}, σ={sigma:.2f}, overlap={ov:.4f} ---")
        iter_start = time.time()

        for rep in range(1, config.n_repeats + 1):
            seed = config.random_seed + rep
            trial_key = f"d{d:.1f}_s{sigma:.2f}_seed{seed}"
            # 该 seed 的数据只生成一次，供所有算法共用
            X_trial, y_trial = generate_overlapping_blobs(seed, d, sigma, config.n_per_cluster)
            data_records[trial_key] = {
                "d": d, "sigma": sigma, "seed": seed,
                "X": X_trial, "y_true": y_trial,
                "predictions": {},
            }

            for alg in algorithms:
                if alg not in calib or calib[alg]['params'] is None:
                    if rep == 1:
                        print(f"    {alg:12s}: SKIPPED (no calibrated params)")
                    continue
                try:
                    row, matlab_runner, _, _, y_pred = run_single_trial(
                        alg, d, sigma, config, seed, calib, matlab_runner,
                        X=X_trial, y_true=y_trial,
                    )
                    row["Repeat"] = rep
                    frames.append(row)
                    data_records[trial_key]["predictions"][alg] = y_pred
                except Exception as e:
                    print(f"    {alg:12s} rep {rep}: ERROR - {e}")

            # 打印该 rep 的结果（按算法汇总需要等所有 rep 完成）

        # 打印每个算法在该 (d, sigma) 下的平均结果
        for alg in algorithms:
            alg_rows = [r for r in frames if r["Algorithm"] == alg and r["d"] == d and r["sigma"] == sigma]
            if alg_rows:
                means = {m: np.mean([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
                stds = {m: np.std([r[m] for r in alg_rows]) for m in METRIC_COLUMNS}
                print(f"    {alg:12s}: "
                      f"ARI={means['ARI']:.4f}±{stds['ARI']:.4f}  "
                      f"FMI={means['FMI']:.4f}±{stds['FMI']:.4f}  "
                      f"F1={means['Macro_F1']:.4f}±{stds['Macro_F1']:.4f}")

        print(f"  (elapsed: {time.time() - iter_start:.1f}s)")

    # 保存数据到磁盘（NPZ格式，紧凑且高效）
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

        # 同时保存一个汇总CSV：每个 (d, sigma, seed) 的元信息
        meta_rows = []
        for trial_key, rec in data_records.items():
            meta_rows.append({
                "trial_key": trial_key,
                "d": rec["d"], "sigma": rec["sigma"], "seed": rec["seed"],
                "n_points": len(rec["X"]),
                "algorithms": ",".join(rec["predictions"].keys()),
            })
        pd.DataFrame(meta_rows).to_csv(data_dir / "dataset_index.csv", index=False)

    return pd.DataFrame(frames), matlab_runner


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
    print(f"\n{'=' * 120}")
    print(title)
    print("=" * 120)
    values = sorted(summary[group_col].unique())
    algs = summary["Algorithm"].unique()
    header = f"{'Algorithm':<14s}"
    for v in values:
        header += f" | {v:>6.2f} ARI"
    print(header)
    print("-" * len(header))
    for alg in algs:
        line = f"{alg:<14s}"
        for v in values:
            row = summary[(summary["Algorithm"] == alg) & (summary[group_col] == v)]
            if row.empty:
                line += f" | {'--':>10s}"
            else:
                line += f" | {row['ARI_mean'].values[0]:>10.4f}"
        print(line)
    print("=" * 120)


def generate_latex_table(summary, group_col, outdir, title, label, filename):
    outdir.mkdir(parents=True, exist_ok=True)
    algs = summary["Algorithm"].unique()
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
    for alg in algs:
        row_line = alg
        for v in values:
            row = summary[(summary["Algorithm"] == alg) & (summary[group_col] == v)]
            if row.empty:
                row_line += " & --"
            else:
                ari = row["ARI_mean"].values[0]
                std = row["ARI_std"].values[0]
                row_line += f" & {ari:.4f}$\\pm${std:.4f}"
        row_line += r" \\"
        lines.append(row_line)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    with open(outdir / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_latex_table_all_metrics(summary, group_col, outdir, title, label, filename):
    outdir.mkdir(parents=True, exist_ok=True)
    algs = summary["Algorithm"].unique()
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
    for alg in algs:
        for i, metric in enumerate(METRIC_COLUMNS):
            if i == 0:
                lines.append(r"\multirow{" + str(n_metrics) + r"}{*}{" + alg + r"}")
            else:
                lines.append("")
            row_line = f" & {metric} "
            for v in values:
                row = summary[(summary["Algorithm"] == alg) & (summary[group_col] == v)]
                if row.empty:
                    row_line += " & --"
                else:
                    row_line += f" & {row[f'{metric}_mean'].values[0]:.4f}"
            row_line += r" \\"
            lines.append(row_line)
        if alg != algs[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    with open(outdir / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 7. Figures
# ============================================================

ALG_COLORS = {
    "IZGC": "#e74c3c", "K-means++": "#3498db", "CLIQUE": "#f39c12",
    "BANG": "#9b59b6", "DBSCAN": "#2ecc71", "GMM": "#e67e22", "DPC-FWSN": "#1abc9c",
}
ALG_MARKERS = {
    "IZGC": "o", "K-means++": "s", "CLIQUE": "D",
    "BANG": "v", "DBSCAN": "^", "GMM": "P", "DPC-FWSN": "X",
}


def plot_degradation_curves(summary, group_col, x_label, outdir, title_prefix):
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(9, 6))
        for alg in summary["Algorithm"].unique():
            sub = summary[summary["Algorithm"] == alg].sort_values(group_col)
            if sub.empty:
                continue
            ax.plot(sub[group_col].astype(float), sub[f"{metric}_mean"],
                    marker=ALG_MARKERS.get(alg, "o"), color=ALG_COLORS.get(alg),
                    linewidth=2, markersize=7, label=alg)
            ax.fill_between(sub[group_col].astype(float),
                            sub[f"{metric}_mean"] - sub[f"{metric}_std"],
                            sub[f"{metric}_mean"] + sub[f"{metric}_std"],
                            alpha=0.10, color=ALG_COLORS.get(alg))
        ax.set_title(f"{title_prefix} ({metric})", fontsize=14)
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(f"{metric} Score", fontsize=13)
        ax.set_ylim(-0.15, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.tight_layout()
        fname = f"{title_prefix.lower().replace(' ', '_')}_{metric.lower()}.png"
        fig.savefig(outdir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_dataset_examples(config, outdir, param_name, param_values):
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(param_values)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    for i, (d, sigma) in enumerate(param_values):
        X, y = generate_overlapping_blobs(config.random_seed, d, sigma, config.n_per_cluster)
        ax = axes[i] if n > 1 else axes
        for lab in np.unique(y):
            mask = y == lab
            ax.scatter(X[mask, 0], X[mask, 1], s=5, alpha=0.5, label=f"C{lab}")
        ov = compute_overlap_ratio(d, sigma)
        ax.set_title(f"d={d:.1f}, σ={sigma:.2f}\n(overlap={ov:.3f})")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Overlapping Blobs ({param_name})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"dataset_examples_{param_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_comparison(config, calib, outdir, param_name, selected_params):
    outdir.mkdir(parents=True, exist_ok=True)
    algs = [a for a in ALGORITHMS if a in calib and calib[a]['params'] is not None]
    matlab_runner = None
    if "DPC-FWSN" in algs and MATLAB_AVAILABLE:
        try:
            matlab_runner = MatlabRunner(method='auto')
        except:
            matlab_runner = None
    n_rows = len(selected_params)
    n_cols = len(algs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    for i, (d, sigma) in enumerate(selected_params):
        seed = config.random_seed + 1
        X, y_true = generate_overlapping_blobs(seed, d, sigma, config.n_per_cluster)
        k_true = len(np.unique(y_true))
        for j, alg in enumerate(algs):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            try:
                _, matlab_runner, _, _, y_pred = run_single_trial(
                    alg, d, sigma, config, seed, calib, matlab_runner,
                    X=X, y_true=y_true,
                )
            except:
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
                ov = compute_overlap_ratio(d, sigma)
                ax.set_ylabel(f"d={d:.1f}, σ={sigma:.2f}\n(ov={ov:.3f})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    if matlab_runner is not None:
        try: matlab_runner.close()
        except: pass
    fig.suptitle(f"Clustering Comparison ({param_name})", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / f"clustering_comparison_{param_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overlap experiment with calibration: 7 algorithms on overlapping Gaussian blobs."
    )
    parser.add_argument("--outdir", type=str, default="results_overlap")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--algorithms", type=str, nargs="+", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-comparison-plot", action="store_true")
    parser.add_argument("--no-dpc", action="store_true")
    parser.add_argument("--calib", type=str, default="results_overlap/calibration_results.json",
                        help="Path to calibration_results.json (default: results_overlap/calibration_results.json)")
    parser.add_argument("--exp", type=str, default="both", choices=["both", "d", "sigma"])
    args = parser.parse_args()

    config = OverlapExperimentConfig(n_repeats=args.repeats)
    algorithms = args.algorithms if args.algorithms else list(ALGORITHMS)
    if args.no_dpc and "DPC-FWSN" in algorithms:
        algorithms = [a for a in algorithms if a != "DPC-FWSN"]
        print("  DPC-FWSN skipped (--no-dpc)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Overlap Experiment (Overlapping Gaussian Blobs)")
    print(f"  Algorithms: {algorithms}")
    print(f"  Repeats: {config.n_repeats} (different Seeds)")
    print(f"  Metrics: {METRIC_COLUMNS}")
    print(f"  Experiments: {args.exp}")
    print("=" * 70)

    # ---- Load calibration results ----
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = Path(args.outdir) / calib_path.name if not calib_path.exists() else calib_path
    if not calib_path.exists():
        # 尝试相对于当前脚本位置的路径
        alt_path = Path(__file__).resolve().parent / calib_path.name
        if alt_path.exists():
            calib_path = alt_path

    if not calib_path.exists():
        print(f"\n  ERROR: calibration file not found: {args.calib}")
        print(f"  Please run overlap_calibration.py first, or specify --calib <path>")
        sys.exit(1)

    print(f"\n  Loading calibration results: {calib_path}")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_raw = json.load(f)

    # 转换为内部格式
    calib = {}
    for alg, info in calib_raw.items():
        calib[alg] = {'params': info.get('params'), 'best_ari': info.get('best_ari', 0.0)}

    print("\n  Calibration parameters:")
    for alg in ALGORITHMS:
        if alg in calib and calib[alg]['params']:
            print(f"    {alg:<14s}: {calib[alg]['params']}  (ARI={calib[alg]['best_ari']:.4f})")
        else:
            print(f"    {alg:<14s}: N/A")

    # Save config
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    # ---- Step 2: Experiments ----
    # Filter algorithms that have calibrated params
    active_algorithms = [a for a in algorithms if a in calib and calib[a]['params'] is not None]
    print(f"\n  Active algorithms (with calibrated params): {active_algorithms}")

    # Init MATLAB
    matlab_runner = None
    if "DPC-FWSN" in active_algorithms and MATLAB_AVAILABLE:
        print("  Initializing MATLAB engine for DPC-FWSN...")
        try:
            matlab_runner = MatlabRunner(method='auto')
            print("  MATLAB engine initialized.")
        except Exception as e:
            print(f"  WARNING: MATLAB init failed: {e}")
            active_algorithms = [a for a in active_algorithms if a != "DPC-FWSN"]

    all_raw_frames = []

    # ---- Experiment 1: Varying d ----
    if args.exp in ("both", "d"):
        print("\n" + "=" * 70)
        print("[Experiment 1] Varying center distance d (σ=1.0 fixed)")
        print(f"  d values: {config.d_values}")
        print("=" * 70)

        d_params = [(d, config.sigma_fixed) for d in config.d_values]

        if not args.no_figures:
            print("\n  Generating dataset visualizations...")
            plot_dataset_examples(config, outdir / "figures", "vary_d", d_params)

        print("\n  Running experiments...")
        raw_d, matlab_runner = run_experiment_group(
            config, active_algorithms, calib, "d", d_params, matlab_runner, outdir
        )
        raw_d.to_csv(outdir / "overlap_exp_d_raw.csv", index=False)
        print(f"\n  Raw: {outdir / 'overlap_exp_d_raw.csv'} ({len(raw_d)} rows)")
        all_raw_frames.append(raw_d)

        summary_d = summarize_results(raw_d, "d")
        summary_d.to_csv(outdir / "overlap_exp_d_summary.csv", index=False)
        print_console_table(summary_d, "d", "Overlap Experiment: Varying d (σ=1.0)")

        if not args.no_figures:
            figdir = outdir / "figures"
            plot_degradation_curves(summary_d, "d", "Center Distance d", figdir, "Overlap Varying d")
            if not args.no_comparison_plot:
                sel_d = [(6.0, config.sigma_fixed), (3.0, config.sigma_fixed), (1.0, config.sigma_fixed)]
                print("  Generating clustering comparison plot...")
                plot_clustering_comparison(config, calib, figdir, "vary_d", sel_d)

        generate_latex_table(summary_d, "d", outdir,
                             "Overlap Robustness: Varying d (ARI)",
                             "tab:overlap_vary_d", "overlap_vary_d_table.tex")
        generate_latex_table_all_metrics(summary_d, "d", outdir,
                                         "Full Overlap Results: Varying d",
                                         "tab:overlap_vary_d_full", "overlap_vary_d_full_table.tex")

    # ---- Experiment 2: Varying σ ----
    if args.exp in ("both", "sigma"):
        print("\n" + "=" * 70)
        print("[Experiment 2] Varying cluster spread σ (d=3.0 fixed)")
        print(f"  σ values: {config.sigma_values}")
        print("=" * 70)

        sigma_params = [(config.d_fixed, s) for s in config.sigma_values]

        if not args.no_figures:
            print("\n  Generating dataset visualizations...")
            plot_dataset_examples(config, outdir / "figures", "vary_sigma", sigma_params)

        print("\n  Running experiments...")
        raw_s, matlab_runner = run_experiment_group(
            config, active_algorithms, calib, "sigma", sigma_params, matlab_runner, outdir
        )
        raw_s.to_csv(outdir / "overlap_exp_sigma_raw.csv", index=False)
        print(f"\n  Raw: {outdir / 'overlap_exp_sigma_raw.csv'} ({len(raw_s)} rows)")
        all_raw_frames.append(raw_s)

        summary_s = summarize_results(raw_s, "sigma")
        summary_s.to_csv(outdir / "overlap_exp_sigma_summary.csv", index=False)
        print_console_table(summary_s, "sigma", "Overlap Experiment: Varying σ (d=3.0)")

        if not args.no_figures:
            figdir = outdir / "figures"
            plot_degradation_curves(summary_s, "sigma", "Cluster Spread σ", figdir, "Overlap Varying sigma")
            if not args.no_comparison_plot:
                sel_s = [(config.d_fixed, 0.5), (config.d_fixed, 1.0), (config.d_fixed, 2.0)]
                print("  Generating clustering comparison plot...")
                plot_clustering_comparison(config, calib, figdir, "vary_sigma", sel_s)

        generate_latex_table(summary_s, "sigma", outdir,
                             "Overlap Robustness: Varying σ (ARI)",
                             "tab:overlap_vary_sigma", "overlap_vary_sigma_table.tex")
        generate_latex_table_all_metrics(summary_s, "sigma", outdir,
                                         "Full Overlap Results: Varying σ",
                                         "tab:overlap_vary_sigma_full", "overlap_vary_sigma_full_table.tex")

    # Close MATLAB
    if matlab_runner is not None:
        try: matlab_runner.close()
        except: pass

    if all_raw_frames:
        all_raw = pd.concat(all_raw_frames, ignore_index=True)
        all_raw.to_csv(outdir / "overlap_exp_all_raw.csv", index=False)
        print(f"\n  All raw: {outdir / 'overlap_exp_all_raw.csv'} ({len(all_raw)} rows)")

    print("\n" + "=" * 70)
    print("Done!")
    print(f"  Output: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
