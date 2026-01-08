"""
ContourBasedClustering 等值线聚类算法
支持参数循环优化
"""

import numpy as np
import sys
import os
from scipy.spatial import cKDTree
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ContourBasedClusteringWithoutZ22 as ContourClustering
from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result
)

# ==================== 配置 ====================
DATASET = "../data/Compound.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'grid_size': (1.0, 5.0, 0.5),           # 从1.0到5.0，步长0.5
    'max_buffer_distance': (0, 5, 1),        # 从0到5，步长1
    'min_cluster_size': (3, 10, 2),          # 从3到10，步长2
    'density_radius': (20, 40, 5)            # 从20到40，步长5
}

# 优化目标: 'ARI', 'AMI', 'FMI', 'Silhouette'
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    import numpy as np
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_contour_clustering(X, grid_size, max_buffer_distance, min_cluster_size, density_radius):
    """运行等值线聚类"""
    tree = cKDTree(X)
    density = np.array([len(tree.query_ball_point(p, density_radius)) for p in X])
    
    combined_data = [(*point, density[i], i) for i, point in enumerate(X)]
    
    cluster = ContourClustering.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=grid_size,
        max_buffer_distance=max_buffer_distance,
        min_cluster_size=min_cluster_size,
        interactive=False
    )
    
    cluster.perform_clustering()
    labels = cluster.get_predicted_labels()
    
    return np.array(labels)


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
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = np.prod([len(v) for v in param_values])
    
    print(f"参数组合总数: {total_combinations}")
    
    results = []
    
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        try:
            labels = run_contour_clustering(X, **params)
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
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("="*60)
    print("ContourBasedClustering 参数优化")
    print("="*60)
    
    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    
    print(f"\n数据集: {dataset_name}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    # 归一化
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")
    
    # 参数搜索
    best_params, best_labels, best_metrics, all_results = grid_search(
        X, y_true, PARAM_RANGES, OPTIMIZE_METRIC
    )
    
    # 打印最佳结果
    print_best_result("ContourBasedClustering", best_params, best_metrics)
    
    # 保存图表
    save_path = f"{OUTPUT_DIR}/ContourBased_{dataset_name}.png"
    plot_clustering_result(
        X, best_labels, 
        f'ContourBased (ARI={best_metrics["ARI"]:.3f})',
        save_path, dpi=300
    )
    
    # 保存真实标签图
    gt_path = f"{OUTPUT_DIR}/GroundTruth_{dataset_name}.png"
    plot_clustering_result(X, y_true, 'Ground Truth', gt_path, dpi=300)
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
