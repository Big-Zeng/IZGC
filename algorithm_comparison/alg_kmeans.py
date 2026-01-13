"""
K-means++ 聚类算法
支持参数循环优化
"""

import numpy as np
import os
from sklearn.cluster import KMeans
from itertools import product

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# ==================== 配置 ====================
DATASET = "../data/s1.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'n_clusters': (15, 15, 1),      # 簇数：从2到15，步长1
    'n_init': (10, 50, 1),        # 初始化次数：从10到50，步长10
    'max_iter': (300, 500, 50)    # 最大迭代：从300到500，步长50
}

# 优化目标
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    import numpy as np
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_kmeans(X, n_clusters, n_init=10, max_iter=300, random_state=42):
    """运行K-means++聚类"""
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm='elkan'
    )
    labels = kmeans.fit_predict(X)
    return labels


def grid_search(X, y_true, param_ranges, optimize_metric='ARI'):
    """网格搜索最佳参数"""
    print("\n开始参数搜索...")
    print(f"优化目标: {optimize_metric}")
    
    # 生成参数值
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
        
        try:
            labels = run_kmeans(X, **params)
            metrics = evaluate_clustering(y_true, labels, X)
            
            score = metrics[optimize_metric]
            results.append({'params': params, 'metrics': metrics, 'labels': labels})
            
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = labels
                best_metrics = metrics
            
            if i % 20 == 0 or i == total_combinations:
                print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                
        except Exception as e:
            pass
    
    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("K-means++ 参数优化")
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
    
    print_best_result("K-means++", best_params, best_metrics)
    
    save_path = f"{OUTPUT_DIR}/KMeans_{dataset_name}.png"
    plot_clustering_result(
        X, best_labels,
        f'K-means++ (ARI={best_metrics["ARI"]:.3f})',
        save_path, dpi=300
    )
    
    # 保存最佳结果到txt
    txt_path = f"{OUTPUT_DIR}/KMeans_{dataset_name}_best.txt"
    save_best_result("K-means++", dataset_name, best_params, best_metrics, txt_path)
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
