"""
运行所有算法对比 - 无监督版本（真实数据）
使用轮廓系数和CH指数作为优化目标
"""

# 禁用matplotlib交互模式
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
import sys
import time
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_comparison.utils import (
    load_dataset, load_real_dataset, normalize_coordinates, 
    evaluate_clustering, evaluate_clustering_unsupervised,
    plot_clustering_result
)
from algorithm_comparison.alg_contour import run_contour_clustering
from algorithm_comparison.alg_kmeans import run_kmeans
from algorithm_comparison.alg_dbscan import run_dbscan
from algorithm_comparison.alg_clique2 import run_clique2
from algorithm_comparison.alg_gmm import run_gmm, COVARIANCE_TYPES
from algorithm_comparison.alg_bang import run_bang, PYCLUSTERING_AVAILABLE as BANG_AVAILABLE

# 尝试导入MATLAB
try:
    from algorithm_comparison.alg_dpc_fwsn import run_dpc_fwsn, MATLAB_AVAILABLE
except:
    MATLAB_AVAILABLE = False

# ==================== 配置 ====================
# 数据集路径（真实数据，无标签）
DATASET = "../data/real_data/长沙市_real.csv"
OUTPUT_DIR = "results_unsupervised"
NORMALIZE = True

# 优化目标: 'Silhouette' 或 'CH'
OPTIMIZE_METRIC = 'Silhouette'

# 各算法参数搜索范围 (start, end, step)
CONTOUR_RANGES = {
    'grid_size': (0.5, 5, 0.001),
    'max_buffer_distance': (0, 0, 1),
    'min_cluster_size': (5, 5, 1),
    'density_radius': (0.01, 0.01, 0.001)
}

KMEANS_RANGES = {
    'n_clusters': (2, 20, 1),
    'n_init': (10, 50, 1),  # 初始化次数：从10到50，步长10
    'max_iter': (300, 500, 50)  # 最大迭代：从300到500，步长50
}

DBSCAN_RANGES = {
    'eps': (1.0, 50, 1.0),  # 邻域半径：从1.0到50.0，步长1.0
    'min_samples': (2, 50, 1)  # 最小样本数：从2到50，步长1
}

CLIQUE_RANGES = {
    'intervals': (4, 100, 1),  # 网格划分数：从4到100，步长1
    'density_threshold': (1, 50, 1)  # 密度阈值：从1到50，步长1
}

GMM_RANGES = {
    'n_components': (2, 20, 1),  # 簇数：从2到15，步长1
    'n_init': (10, 50, 1)  # 初始化次数：从5到20，步长5
}

BANG_RANGES = {
    'levels': (4, 50, 1),  # 网格层级：从4到100，步长1
    'density_threshold': (0.01, 1, 0.05)  # 密度阈值：从0.01到1，步长0.05
}

DPC_RANGES = {
    'k': (1, 50, 1)  # 从1到50，步长1
}

RUN_MATLAB = True
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def optimize_algorithm_unsupervised(name, run_func, X, param_ranges, optimize_metric='Silhouette', **extra_args):
    """无监督算法参数优化"""
    print(f"\n{'='*60}")
    print(f"优化 {name} (无监督)")
    print(f"{'='*60}")
    
    # 生成参数值
    param_grid = {}
    for pname, (start, end, step) in param_ranges.items():
        param_grid[pname] = generate_param_values(start, end, step)
        print(f"  {pname}: {start} -> {end}, 步长={step}, 共{len(param_grid[pname])}个值")
    
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_metrics = None
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total = np.prod([len(v) for v in param_values])
    
    print(f"参数组合数: {int(total)}")
    print(f"优化目标: {optimize_metric}")
    
    start_time = time.time()
    
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        try:
            labels = run_func(X, **params, **extra_args)
            if labels is None:
                continue
            
            metrics = evaluate_clustering_unsupervised(labels, X)
            
            if metrics['n_clusters'] < 2:
                continue
            
            score = metrics[optimize_metric]
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_labels = labels.copy() if isinstance(labels, np.ndarray) else np.array(labels)
                best_metrics = metrics.copy()
                
            if i % 50 == 0 or i == total:
                print(f"  进度: {i}/{int(total)}, 当前最佳{optimize_metric}: {best_score:.4f}")
                
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    
    if best_params:
        print(f"\n最佳参数: {best_params}")
        print(f"最佳{optimize_metric}: {best_score:.4f}")
        print(f"CH: {best_metrics['CH']:.2f}")
        print(f"簇数: {best_metrics['n_clusters']}, 噪声: {best_metrics['n_noise']}")
        print(f"用时: {elapsed:.2f}s")
    else:
        print(f"\n未找到有效参数")
    
    return {
        'name': name,
        'params': best_params,
        'labels': best_labels,
        'metrics': best_metrics,
        'time': elapsed
    }


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("聚类算法综合对比 - 无监督版本（真实数据）")
    print(f"优化目标: {OPTIMIZE_METRIC}")
    print("="*70)
    
    # 加载数据
    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    
    print(f"\n数据集: {dataset_name}")
    print(f"样本数: {len(X)}")
    if y_true is not None:
        print(f"真实簇数: {len(np.unique(y_true))}")
    else:
        print("无真实标签（无监督模式）")
    
    X_original = X.copy()
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")
    
    results = []
    
    # 1. ContourBasedClustering
    result = optimize_algorithm_unsupervised(
        "ContourBased", run_contour_clustering, X, 
        CONTOUR_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"ContourBased (Sil={result['metrics']['Silhouette']:.3f})",
            f"{OUTPUT_DIR}/01_ContourBased_{dataset_name}.png", dpi=300
        )
    
    # 2. K-means++
    result = optimize_algorithm_unsupervised(
        "K-means++", run_kmeans, X,
        KMEANS_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"K-means++ (Sil={result['metrics']['Silhouette']:.3f})",
            f"{OUTPUT_DIR}/02_KMeans_{dataset_name}.png", dpi=300
        )
    
    # 3. DBSCAN
    result = optimize_algorithm_unsupervised(
        "DBSCAN", run_dbscan, X,
        DBSCAN_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"DBSCAN (Sil={result['metrics']['Silhouette']:.3f})",
            f"{OUTPUT_DIR}/03_DBSCAN_{dataset_name}.png", dpi=300
        )
    
    # 4. CLIQUE
    result = optimize_algorithm_unsupervised(
        "CLIQUE", run_clique2, X,
        CLIQUE_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"CLIQUE (Sil={result['metrics']['Silhouette']:.3f})",
            f"{OUTPUT_DIR}/04_CLIQUE_{dataset_name}.png", dpi=300
        )
    
    # 5. GMM
    print(f"\n{'='*60}")
    print("优化 GMM (无监督)")
    print(f"{'='*60}")
    
    gmm_param_grid = {}
    for pname, (start, end, step) in GMM_RANGES.items():
        gmm_param_grid[pname] = generate_param_values(start, end, step)
        print(f"  {pname}: {start} -> {end}, 步长={step}, 共{len(gmm_param_grid[pname])}个值")
    print(f"  covariance_type: {COVARIANCE_TYPES}")
    
    best_gmm_score = -np.inf
    best_gmm_params = None
    best_gmm_labels = None
    best_gmm_metrics = None
    
    for n_components in gmm_param_grid['n_components']:
        for n_init in gmm_param_grid['n_init']:
            for cov_type in COVARIANCE_TYPES:
                try:
                    labels = run_gmm(X, n_components, cov_type, n_init)
                    if labels is None:
                        continue
                    metrics = evaluate_clustering_unsupervised(labels, X)
                    if metrics['n_clusters'] < 2:
                        continue
                    score = metrics[OPTIMIZE_METRIC]
                    if score > best_gmm_score:
                        best_gmm_score = score
                        best_gmm_params = {'n_components': n_components, 'covariance_type': cov_type, 'n_init': n_init}
                        best_gmm_labels = labels.copy()
                        best_gmm_metrics = metrics.copy()
                except:
                    pass
    
    if best_gmm_params:
        print(f"\n最佳参数: {best_gmm_params}")
        print(f"最佳{OPTIMIZE_METRIC}: {best_gmm_score:.4f}")
        results.append({
            'name': 'GMM',
            'params': best_gmm_params,
            'labels': best_gmm_labels,
            'metrics': best_gmm_metrics,
            'time': 0
        })
        plot_clustering_result(
            X, best_gmm_labels,
            f"GMM (Sil={best_gmm_metrics['Silhouette']:.3f})",
            f"{OUTPUT_DIR}/05_GMM_{dataset_name}.png", dpi=300
        )
    
    # 6. BANG
    if BANG_AVAILABLE:
        result = optimize_algorithm_unsupervised(
            "BANG", run_bang, X,
            BANG_RANGES, OPTIMIZE_METRIC
        )
        if result['labels'] is not None:
            results.append(result)
            plot_clustering_result(
                X, result['labels'],
                f"BANG (Sil={result['metrics']['Silhouette']:.3f})",
                f"{OUTPUT_DIR}/06_BANG_{dataset_name}.png", dpi=300
            )
    
    # 7. DPC-FWSN (MATLAB)
    if RUN_MATLAB and MATLAB_AVAILABLE:
        print(f"\n{'='*60}")
        print("优化 DPC-FWSN (MATLAB) - 无监督")
        print(f"{'='*60}")
        
        k_values = generate_param_values(*DPC_RANGES['k'])
        print(f"  k: {DPC_RANGES['k'][0]} -> {DPC_RANGES['k'][1]}, 步长={DPC_RANGES['k'][2]}, 共{len(k_values)}个值")
        
        try:
            from matlab_integration import MatlabRunner
            matlab_runner = MatlabRunner(method='auto')
            
            best_score = -np.inf
            best_params = None
            best_labels = None
            best_metrics = None
            
            # 无监督模式：假设簇数为数据量的平方根
            n_clusters = max(2, int(np.sqrt(len(X) / 10)))
            # 创建假的标签用于MATLAB接口
            fake_labels = np.zeros(len(X))
            data_with_labels = np.column_stack([X, fake_labels])
            
            for k in k_values:
                try:
                    labels, _ = matlab_runner.run_algorithm_pr(
                        data_with_labels, n_clusters, k
                    )
                    if labels is not None:
                        eval_metrics = evaluate_clustering_unsupervised(labels, X)
                        if eval_metrics['n_clusters'] < 2:
                            continue
                        score = eval_metrics[OPTIMIZE_METRIC]
                        print(f"  k={k}: {OPTIMIZE_METRIC}={score:.4f}")
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'k': k}
                            best_labels = labels.copy()
                            best_metrics = eval_metrics.copy()
                except Exception as e:
                    print(f"  k={k}: 失败 - {e}")
            
            matlab_runner.close()
            
            if best_labels is not None:
                results.append({
                    'name': 'DPC-FWSN',
                    'params': best_params,
                    'labels': best_labels,
                    'metrics': best_metrics,
                    'time': 0
                })
                plot_clustering_result(
                    X, best_labels,
                    f"DPC-FWSN (Sil={best_metrics['Silhouette']:.3f})",
                    f"{OUTPUT_DIR}/07_DPC-FWSN_{dataset_name}.png", dpi=300
                )
        except Exception as e:
            print(f"DPC-FWSN失败: {e}")
    
    # 打印汇总
    print("\n" + "="*90)
    print("算法对比汇总（无监督）")
    print("="*90)
    
    header = f"{'算法':<15} {'Silhouette':<12} {'CH':<15} {'簇数':<8} {'噪声':<8}"
    print(header)
    print("-"*90)
    
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<15} {m['Silhouette']:<12.4f} {m['CH']:<15.2f} "
              f"{m['n_clusters']:<8} {m['n_noise']:<8}")
    
    print("="*90)
    
    # 保存结果到CSV
    csv_data = []
    for r in results:
        m = r['metrics']
        csv_data.append({
            'Algorithm': r['name'],
            'Best_Params': str(r['params']),
            'Silhouette': m['Silhouette'],
            'CH': m['CH'],
            'n_clusters': m['n_clusters'],
            'n_noise': m['n_noise']
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f"{OUTPUT_DIR}/comparison_results_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存: {csv_path}")
    
    # 保存每个算法的最佳结果到txt
    for r in results:
        txt_path = f"{OUTPUT_DIR}/{r['name']}_{dataset_name}_best.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"算法: {r['name']}\n")
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"最佳参数: {r['params']}\n")
            f.write(f"\n评估指标（无监督）:\n")
            f.write(f"  Silhouette: {r['metrics']['Silhouette']:.4f}\n")
            f.write(f"  CH: {r['metrics']['CH']:.2f}\n")
            f.write(f"  簇数: {r['metrics']['n_clusters']}\n")
            f.write(f"  噪声点: {r['metrics']['n_noise']}\n")
        print(f"  已保存: {txt_path}")
    
    print(f"\n图表已保存到: {OUTPUT_DIR}/")
    print("\n完成！")


if __name__ == "__main__":
    main()
