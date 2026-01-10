"""
GMM - 高斯混合模型聚类算法
使用sklearn的GaussianMixture
支持参数循环优化
"""

import numpy as np
import os
from itertools import product
from sklearn.mixture import GaussianMixture

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# ==================== 配置 ====================
DATASET = "../data/Compound.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# 参数搜索范围 (start, end, step)
# GaussianMixture参数:
# - n_components: 高斯分量数（簇数）
# - covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
# - n_init: 初始化次数
PARAM_RANGES = {
    'n_components': (2, 15, 1),    # 簇数：从2到15，步长1
    'n_init': (1, 20, 5)           # 初始化次数：从5到20，步长5
}

# 协方差类型列表（单独处理）
COVARIANCE_TYPES = ['full', 'tied', 'diag', 'spherical']

# 优化目标
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_gmm(X, n_components, covariance_type, n_init):
    """
    运行高斯混合模型聚类
    
    参数:
        X: 数据点数组
        n_components: 高斯分量数（簇数）
        covariance_type: 协方差类型
        n_init: 初始化次数
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=42,
        max_iter=200
    )
    
    labels = gmm.fit_predict(X)
    return labels


def grid_search(X, y_true, param_ranges, covariance_types, optimize_metric='ARI'):
    """网格搜索最佳参数"""
    print("\n开始参数搜索...")
    print(f"优化目标: {optimize_metric}")
    
    # 生成参数值
    param_grid = {}
    for name, (start, end, step) in param_ranges.items():
        param_grid[name] = generate_param_values(start, end, step)
        print(f"  {name}: {start} -> {end}, 步长={step}, 共{len(param_grid[name])}个值")
    
    print(f"  covariance_type: {covariance_types}")
    
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_metrics = None
    
    # 计算总组合数
    total_combinations = (len(param_grid['n_components']) * 
                         len(param_grid['n_init']) * 
                         len(covariance_types))
    
    print(f"参数组合总数: {total_combinations}")
    
    results = []
    i = 0
    
    for n_components in param_grid['n_components']:
        for n_init in param_grid['n_init']:
            for cov_type in covariance_types:
                i += 1
                params = {
                    'n_components': n_components,
                    'covariance_type': cov_type,
                    'n_init': n_init
                }
                
                try:
                    labels = run_gmm(X, **params)
                    metrics = evaluate_clustering(y_true, labels, X)
                    
                    if metrics['n_clusters'] < 2:
                        if i % 20 == 0 or i == total_combinations:
                            print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                        continue
                    
                    score = metrics[optimize_metric]
                    results.append({'params': params, 'metrics': metrics, 'labels': labels})
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_labels = labels.copy()
                        best_metrics = metrics.copy()
                    
                    if i % 20 == 0 or i == total_combinations:
                        print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                        
                except Exception as e:
                    if i % 20 == 0 or i == total_combinations:
                        print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
    
    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("GMM (高斯混合模型) 参数优化")
    print("="*60)
    
    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    
    print(f"\n数据集: {dataset_name}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")
    
    best_params, best_labels, best_metrics, all_results = grid_search(
        X, y_true, PARAM_RANGES, COVARIANCE_TYPES, OPTIMIZE_METRIC
    )
    
    if best_params:
        print_best_result("GMM", best_params, best_metrics)
        
        save_path = f"{OUTPUT_DIR}/GMM_{dataset_name}.png"
        plot_clustering_result(
            X, best_labels,
            f'GMM (ARI={best_metrics["ARI"]:.3f})',
            save_path, dpi=300
        )
        
        # 保存最佳结果到txt
        txt_path = f"{OUTPUT_DIR}/GMM_{dataset_name}_best.txt"
        save_best_result("GMM", dataset_name, best_params, best_metrics, txt_path)
    else:
        print("\n未找到有效的参数组合")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
