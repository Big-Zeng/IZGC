"""
重叠实验参数校准脚本
对初始分离数据集 (d=6, σ=1.0, overlap≈0.003) 上对7种算法进行参数网格搜索，
以ARI为优化目标，求得各算法最优参数，保存为 JSON 供 overlap_experiment.py 使用。

算法: IZGC, K-means++, GMM, DBSCAN, CLIQUE, BANG, DPC-FWSN
数据: 两个 Gaussian cluster, μ₁=(-3,0), μ₂=(3,0), σ=1.0, 每簇500点, 归一化到[0,100]

用法:
  python overlap_calibration.py
  python overlap_calibration.py --no-dpc
  python overlap_calibration.py --outdir results_overlap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

# 项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustering_metrics import ClusteringMetrics

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

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
# 配置
# ============================================================

RANDOM_SEED = 20260825

# 初始数据集: d=6, σ=1.0 (几乎完全分离, overlap≈0.003)
CALIB_D = 6.0
CALIB_SIGMA = 1.0
N_PER_CLUSTER = 500

ALGORITHMS = ["IZGC", "K-means++", "GMM", "DBSCAN", "CLIQUE", "BANG", "DPC-FWSN"]

# 各算法参数搜索范围
IZGC_RANGES = {
    'grid_size': [round(v, 1) for v in np.arange(1.2, 1.6, 0.01)],
    'R': [20, 25, 30, 35, 40],
    'theta': [3, 5, 7, 10],
}
KMEANS_RANGES = {
    'n_clusters': [2],   # 已知2簇
    'n_init': [10, 20, 30, 50],
    'max_iter': [300, 500, 1000],
}
GMM_COV_TYPES = ['full', 'tied', 'diag', 'spherical']
GMM_RANGES = {
    'n_components': [2],
    'n_init': [5, 10, 20],
}
DBSCAN_RANGES = {
    'eps': [round(v, 1) for v in np.arange(1.0, 50.1, 1.0)],
    'min_samples': list(range(2, 21)),
}
CLIQUE_RANGES = {
    'intervals': list(range(4, 51)),
    'density_threshold': list(range(1, 31)),
}
BANG_RANGES = {
    'levels': list(range(4, 31)),
    'density_threshold': [round(v, 2) for v in np.arange(0.01, 1.01, 0.05)],
}
DPC_RANGES = {
    'k': list(range(1, 51)),
}


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


def compute_overlap_ratio(d: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return float(2 * norm.cdf(-d / (2 * sigma)))


def generate_overlapping_blobs(
    seed: int,
    d: float,
    sigma: float = 1.0,
    n_per_cluster: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
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


def quick_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """快速ARI评估"""
    return float(ClusteringMetrics.evaluate_all(y_true, y_pred)["ARI"])


def n_clusters(labels: np.ndarray) -> int:
    return len(set(labels)) - (1 if -1 in labels else 0)


# ============================================================
# 算法运行函数
# ============================================================

def run_izgc(X, grid_size, R, theta, merge_small_clusters=True):
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


def run_gmm(X, n_components, covariance_type='full', n_init=10, random_state=42):
    return GaussianMixture(n_components=n_components,
                           covariance_type=covariance_type,
                           n_init=n_init, random_state=random_state,
                           max_iter=200).fit_predict(X)


def run_dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples,
                  metric='euclidean').fit_predict(X)


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
# 各算法网格搜索
# ============================================================

def calibrate_izgc(X, y_true) -> Dict:
    print("\n  [IZGC] Calibrating grid_size, R, theta...")
    ranges = IZGC_RANGES
    total = len(ranges['grid_size']) * len(ranges['R']) * len(ranges['theta'])
    print(f"    组合数: {total}")
    print(f"    grid_size: {ranges['grid_size'][0]}~{ranges['grid_size'][-1]} (步长0.1, {len(ranges['grid_size'])}个)")
    print(f"    R: {ranges['R']}")
    print(f"    theta: {ranges['theta']}")

    best_ari = -np.inf
    best_params = None
    count = 0
    t0 = time.time()

    for gs in ranges['grid_size']:
        for R in ranges['R']:
            for th in ranges['theta']:
                count += 1
                try:
                    labels = run_izgc(X, gs, R, th)
                    if n_clusters(labels) < 2:
                        continue
                    ari = quick_ari(y_true, labels)
                    if ari > best_ari:
                        best_ari = ari
                        best_params = {'grid_size': gs, 'R': R, 'theta': th}
                except:
                    pass
        if count % 20 == 0:
            print(f"    进度: {count}/{total}, best ARI={best_ari:.4f}")

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => IZGC best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result


def calibrate_kmeans(X, y_true) -> Dict:
    print("\n  [K-means++] Calibrating n_init, max_iter...")
    ranges = KMEANS_RANGES
    total = len(ranges['n_init']) * len(ranges['max_iter'])
    print(f"    组合数: {total}")

    best_ari = -np.inf
    best_params = None
    t0 = time.time()

    for ni in ranges['n_init']:
        for mi in ranges['max_iter']:
            try:
                labels = run_kmeans(X, n_clusters=2, n_init=ni, max_iter=mi,
                                    random_state=RANDOM_SEED)
                ari = quick_ari(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'n_clusters': 2, 'n_init': ni, 'max_iter': mi}
            except:
                pass

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => K-means++ best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result


def calibrate_gmm(X, y_true) -> Dict:
    print("\n  [GMM] Calibrating covariance_type, n_init...")
    total = len(GMM_COV_TYPES) * len(GMM_RANGES['n_init'])
    print(f"    组合数: {total}")

    best_ari = -np.inf
    best_params = None
    t0 = time.time()

    for cov in GMM_COV_TYPES:
        for ni in GMM_RANGES['n_init']:
            try:
                labels = run_gmm(X, n_components=2, covariance_type=cov,
                                 n_init=ni, random_state=RANDOM_SEED)
                ari = quick_ari(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'n_components': 2, 'covariance_type': cov, 'n_init': ni}
            except:
                pass

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => GMM best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result


def calibrate_dbscan(X, y_true) -> Dict:
    print("\n  [DBSCAN] Calibrating eps, min_samples...")
    ranges = DBSCAN_RANGES
    total = len(ranges['eps']) * len(ranges['min_samples'])
    print(f"    组合数: {total}")
    print(f"    eps: {ranges['eps'][0]}~{ranges['eps'][-1]} (步长1.0, {len(ranges['eps'])}个)")
    print(f"    min_samples: {ranges['min_samples'][0]}~{ranges['min_samples'][-1]} (步长1, {len(ranges['min_samples'])}个)")

    best_ari = -np.inf
    best_params = None
    count = 0
    t0 = time.time()

    for eps in ranges['eps']:
        for ms in ranges['min_samples']:
            count += 1
            try:
                labels = run_dbscan(X, eps, ms)
                if n_clusters(labels) < 2:
                    continue
                ari = quick_ari(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'eps': eps, 'min_samples': ms}
            except:
                pass
        if count % 50 == 0:
            print(f"    进度: {count}/{total}, best ARI={best_ari:.4f}")

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => DBSCAN best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result


def calibrate_clique(X, y_true) -> Dict:
    if not PYCLUSTERING_AVAILABLE:
        print("\n  [CLIQUE] SKIPPED (pyclustering not available)")
        return {'params': None, 'best_ari': -1.0, 'time_s': 0}

    print("\n  [CLIQUE] Calibrating intervals, density_threshold...")
    ranges = CLIQUE_RANGES
    total = len(ranges['intervals']) * len(ranges['density_threshold'])
    print(f"    组合数: {total}")
    print(f"    intervals: {ranges['intervals'][0]}~{ranges['intervals'][-1]} ({len(ranges['intervals'])}个)")
    print(f"    density_threshold: {ranges['density_threshold'][0]}~{ranges['density_threshold'][-1]} ({len(ranges['density_threshold'])}个)")

    best_ari = -np.inf
    best_params = None
    count = 0
    t0 = time.time()

    for iv in ranges['intervals']:
        for dt in ranges['density_threshold']:
            count += 1
            try:
                labels = run_clique(X, iv, dt)
                if n_clusters(labels) < 2:
                    continue
                ari = quick_ari(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'intervals': iv, 'density_threshold': dt}
            except:
                pass
        if count % 50 == 0:
            print(f"    进度: {count}/{total}, best ARI={best_ari:.4f}")

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => CLIQUE best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result


def calibrate_bang(X, y_true) -> Dict:
    if not PYCLUSTERING_AVAILABLE:
        print("\n  [BANG] SKIPPED (pyclustering not available)")
        return {'params': None, 'best_ari': -1.0, 'time_s': 0}

    print("\n  [BANG] Calibrating levels, density_threshold...")
    ranges = BANG_RANGES
    total = len(ranges['levels']) * len(ranges['density_threshold'])
    print(f"    组合数: {total}")
    print(f"    levels: {ranges['levels'][0]}~{ranges['levels'][-1]} ({len(ranges['levels'])}个)")
    print(f"    density_threshold: {ranges['density_threshold'][0]}~{ranges['density_threshold'][-1]} (步长0.05, {len(ranges['density_threshold'])}个)")

    best_ari = -np.inf
    best_params = None
    count = 0
    t0 = time.time()

    for lv in ranges['levels']:
        for dt in ranges['density_threshold']:
            count += 1
            try:
                labels = run_bang(X, lv, dt)
                if n_clusters(labels) < 2:
                    continue
                ari = quick_ari(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'levels': lv, 'density_threshold': dt}
            except:
                pass
        if count % 30 == 0:
            print(f"    进度: {count}/{total}, best ARI={best_ari:.4f}")

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => BANG best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result


def calibrate_dpc_fwsn(X, y_true, matlab_runner=None) -> Tuple[Dict, Optional[object]]:
    if not MATLAB_AVAILABLE:
        print("\n  [DPC-FWSN] SKIPPED (matlab_integration not available)")
        return {'params': None, 'best_ari': -1.0, 'time_s': 0}, matlab_runner

    print("\n  [DPC-FWSN] Calibrating k (1~50)...")
    k_values = DPC_RANGES['k']
    print(f"    k: {k_values[0]}~{k_values[-1]} (步长1, {len(k_values)}个)")

    best_ari = -np.inf
    best_params = None
    t0 = time.time()

    for k in k_values:
        try:
            labels, matlab_runner = run_dpc_fwsn(X, k, 2, matlab_runner)
            ari = quick_ari(y_true, labels)
            print(f"    k={k}: ARI={ari:.4f}")
            if ari > best_ari:
                best_ari = ari
                best_params = {'k': k}
        except Exception as e:
            print(f"    k={k}: ERROR - {e}")

    elapsed = time.time() - t0
    result = {'params': best_params, 'best_ari': round(best_ari, 4), 'time_s': round(elapsed, 1)}
    print(f"    => DPC-FWSN best: {best_params} ARI={best_ari:.4f} ({elapsed:.1f}s)")
    return result, matlab_runner


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overlap experiment parameter calibration: grid-search on initial well-separated data."
    )
    parser.add_argument("--outdir", type=str, default="results_overlap")
    parser.add_argument("--no-dpc", action="store_true", help="Skip DPC-FWSN")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                        help="Algorithms to calibrate (default: all 7)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    algorithms = args.algorithms if args.algorithms else list(ALGORITHMS)
    if args.no_dpc and "DPC-FWSN" in algorithms:
        algorithms = [a for a in algorithms if a != "DPC-FWSN"]
        print("  DPC-FWSN skipped (--no-dpc)")

    print("=" * 70)
    print("Overlap Experiment - Parameter Calibration")
    print("=" * 70)
    print(f"  Data: d={CALIB_D}, σ={CALIB_SIGMA}, n_per_cluster={N_PER_CLUSTER}")
    print(f"  Overlap ratio: {compute_overlap_ratio(CALIB_D, CALIB_SIGMA):.4f}")
    print(f"  Seed: {RANDOM_SEED}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Optimization target: ARI")
    print(f"  Output: {outdir}")
    print("=" * 70)

    # 1. 生成校准数据集
    print("\n[1] Generating calibration dataset...")
    X, y_true = generate_overlapping_blobs(RANDOM_SEED, CALIB_D, CALIB_SIGMA, N_PER_CLUSTER)
    print(f"  X: {X.shape}, y: {y_true.shape}, labels: {np.unique(y_true)}")
    print(f"  X range: X=[{X[:,0].min():.2f}, {X[:,0].max():.2f}], "
          f"Y=[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")

    # 2. 逐算法网格搜索
    print("\n[2] Running grid search for each algorithm...")
    results = {}
    matlab_runner = None

    if "IZGC" in algorithms:
        results["IZGC"] = calibrate_izgc(X, y_true)

    if "K-means++" in algorithms:
        results["K-means++"] = calibrate_kmeans(X, y_true)

    if "GMM" in algorithms:
        results["GMM"] = calibrate_gmm(X, y_true)

    if "DBSCAN" in algorithms:
        results["DBSCAN"] = calibrate_dbscan(X, y_true)

    if "CLIQUE" in algorithms:
        results["CLIQUE"] = calibrate_clique(X, y_true)

    if "BANG" in algorithms:
        results["BANG"] = calibrate_bang(X, y_true)

    if "DPC-FWSN" in algorithms:
        dpc_result, matlab_runner = calibrate_dpc_fwsn(X, y_true, matlab_runner)
        results["DPC-FWSN"] = dpc_result

    # 关闭MATLAB
    if matlab_runner is not None:
        try:
            matlab_runner.close()
        except:
            pass

    # 3. 汇总打印
    print("\n" + "=" * 70)
    print("Calibration Summary")
    print("=" * 70)
    print(f"  Dataset: d={CALIB_D}, σ={CALIB_SIGMA}, "
          f"overlap={compute_overlap_ratio(CALIB_D, CALIB_SIGMA):.4f}")
    print()
    header = f"{'Algorithm':<14s} {'ARI':<10s} {'Time(s)':<10s} {'Best Params'}"
    print(header)
    print("-" * 70)
    for alg in ALGORITHMS:
        if alg in results and results[alg]['params']:
            r = results[alg]
            print(f"{alg:<14s} {r['best_ari']:<10.4f} {r['time_s']:<10.1f} {r['params']}")
        elif alg in results:
            print(f"{alg:<14s} {'N/A':<10s} {'N/A':<10s} (no valid params found)")
    print("=" * 70)

    # 4. 保存JSON
    json_path = outdir / "calibration_results.json"
    save_data = {}
    for alg, r in results.items():
        save_data[alg] = {
            'params': r['params'],
            'best_ari': r['best_ari'],
            'time_s': r['time_s'],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Calibration results saved: {json_path}")

    # 5. 保存CSV
    csv_path = outdir / "calibration_results.csv"
    csv_rows = []
    for alg, r in results.items():
        csv_rows.append({
            'Algorithm': alg,
            'Best_ARI': r['best_ari'],
            'Time_s': r['time_s'],
            'Params': json.dumps(r['params'], ensure_ascii=False),
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV saved: {csv_path}")

    # 6. 保存可读txt
    txt_path = outdir / "calibration_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Overlap Experiment - Parameter Calibration Results\n")
        f.write(f"Dataset: d={CALIB_D}, sigma={CALIB_SIGMA}, "
                f"overlap={compute_overlap_ratio(CALIB_D, CALIB_SIGMA):.4f}\n")
        f.write(f"Seed: {RANDOM_SEED}\n")
        f.write(f"Optimization target: ARI\n")
        f.write("=" * 70 + "\n\n")
        for alg in ALGORITHMS:
            if alg in results and results[alg]['params']:
                r = results[alg]
                f.write(f"[{alg}]\n")
                f.write(f"  Best ARI: {r['best_ari']:.4f}\n")
                f.write(f"  Best Params: {r['params']}\n")
                f.write(f"  Time: {r['time_s']:.1f}s\n\n")
            elif alg in results:
                f.write(f"[{alg}]\n  No valid params found.\n\n")
    print(f"  TXT saved: {txt_path}")

    print("\n" + "=" * 70)
    print("Calibration complete!")
    print(f"  Use {json_path} as input for overlap_experiment.py (--skip-calibration)")
    print("=" * 70)


if __name__ == "__main__":
    main()
