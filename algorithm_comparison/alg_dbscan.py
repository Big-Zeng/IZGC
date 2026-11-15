"""
DBSCAN 聚类算法
支持参数循环优化
"""

import numpy as np
import os
from sklearn.cluster import DBSCAN
from itertools import product

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# ==================== Configuration ====================
DATASET = "../data/real_data/长沙市_real.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# Parameter search range (start, end, step)
PARAM_RANGES = {
    'eps': (1.0, 50, 1.0),        # Neighbourhood radius: 1.0 to 15.0 in steps of 1.0
    'min_samples': (2, 50, 1)       # Minimum samples: 3 to 15 in steps of 2
}

# Optimisation objective
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    import numpy as np
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_dbscan(X, eps, min_samples):
    """运行DBSCAN聚类"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(X)
    return labels


def grid_search(X, y_true, param_ranges, optimize_metric='ARI'):
    """网格搜索最佳参数"""
    print("\n开始参数搜索...")
    print(f"优化目标: {optimize_metric}")
    
    # Generate parameter values
    param_grid = {}
    for name, (start, end, step) in param_ranges.items():
        param_grid[name] = generate_param_values(start, end, step)
        print(f"  {name}: {start} -> {end}, 步长={step}, 共{len(param_grid[name])}个值")
    
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_metrics = None
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = np.prod([len(v) for v in param_values])
    
    print(f"参数组合总数: {total_combinations}")
    
    results = []
    
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        # Report progress every 20 trials and on the last one
        if i % 20 == 0 or i == total_combinations:
            print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
        
        try:
            labels = run_dbscan(X, **params)
            metrics = evaluate_clustering(y_true, labels, X)
            
            # Skip the cases with a single cluster or nothing but noise
            if metrics['n_clusters'] < 2:
                continue
            
            score = metrics[optimize_metric]
            results.append({'params': params, 'metrics': metrics, 'labels': labels})
            
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = labels
                best_metrics = metrics
                
        except Exception as e:
            pass
    
    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("DBSCAN 参数优化")
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
        X, y_true, PARAM_RANGES, OPTIMIZE_METRIC
    )
    
    if best_params:
        print_best_result("DBSCAN", best_params, best_metrics)
        
        save_path = f"{OUTPUT_DIR}/DBSCAN_{dataset_name}.png"
        plot_clustering_result(
            X, best_labels,
            f'DBSCAN (ARI={best_metrics["ARI"]:.3f})',
            save_path, dpi=300
        )
        
        # Save the best result to txt
        txt_path = f"{OUTPUT_DIR}/DBSCAN_{dataset_name}_best.txt"
        save_best_result("DBSCAN", dataset_name, best_params, best_metrics, txt_path)
    else:
        print("\n未找到有效的参数组合")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
