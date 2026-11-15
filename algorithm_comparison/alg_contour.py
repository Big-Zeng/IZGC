"""
ContourBasedClustering 等值线聚类算法
支持参数循环优化
"""

# The Agg backend must be set before importing ContourBasedClusteringWithoutZ22
# The module sets TkAgg internally
import matplotlib
matplotlib.use('Agg')

import numpy as np
import sys
import os
from scipy.spatial import cKDTree
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ContourBasedClusteringWithoutZ22 as ContourClustering
from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# ==================== Configuration ====================
DATASET = "../data/2d_3c.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# Parameter search range (start, end, step)
PARAM_RANGES = {
    'grid_size': (1.742,  1.742, 0.001),           # From 1.0 to 5.0 in steps of 0.5
    'max_buffer_distance': (0, 0, 1),        # From 0 to 5 in steps of 1
    'min_cluster_size': (5, 5, 1),          # From 3 to 10 in steps of 2
    'density_radius': (30, 30, 1)            # From 20 to 40 in steps of 5
}

# Objective: 'ARI', 'AMI', 'FMI' or 'Silhouette'
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    import numpy as np
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_contour_clustering(X, grid_size, max_buffer_distance, min_cluster_size, density_radius,merge_small_clusters = True):
    #grid_size truncated to three decimals

    grid_size = '%.3f' % grid_size
    grid_size = float(grid_size)
    print('grid_size:', grid_size)
   
    """运行等值线聚类"""
    from scipy.spatial import Voronoi
    matplotlib.use('Agg')
    # Compute density
    tree = cKDTree(X)
    density = np.array([len(tree.query_ball_point(p, density_radius)) for p in X])

    # Prepare data as (x, y, density, index)
    combined_data = [(*point, density[i], i) for i, point in enumerate(X)]

    # Create clustering instance
    cluster = ContourClustering.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=grid_size,
        max_buffer_distance=max_buffer_distance,
        min_cluster_size=min_cluster_size,
        merge_small_clusters=merge_small_clusters

    )

    # Run clustering
    cluster.perform_clustering()

    # Get labels
    labels = cluster.get_predicted_labels()

    return np.array(labels)


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

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = int(np.prod([len(v) for v in param_values]))

    print(f"参数组合总数: {total_combinations}")

    results = []
    success_count = 0
    fail_count = 0

    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))

        try:
            labels = run_contour_clustering(X, **params)

            # Check whether the clustering is valid
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                fail_count += 1
                if i % 50 == 0 or i == total_combinations:
                    print(f"  进度: {i}/{total_combinations}, 成功: {success_count}, 当前最佳{optimize_metric}: {best_score:.4f}")
                continue

            metrics = evaluate_clustering(y_true, labels, X)

            score = metrics[optimize_metric]
            results.append({'params': params, 'metrics': metrics, 'labels': labels})
            success_count += 1

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_labels = labels.copy()
                best_metrics = metrics.copy()

            if i % 50 == 0 or i == total_combinations:
                print(f"  进度: {i}/{total_combinations}, 成功: {success_count}, 当前最佳{optimize_metric}: {best_score:.4f}")

        except Exception as e:
            fail_count += 1
            if i % 100 == 0:
                print(f"  进度: {i}/{total_combinations}, 参数 {params} 失败: {str(e)[:50]}")

    print(f"\n搜索完成: 成功 {success_count}/{total_combinations}, 失败 {fail_count}/{total_combinations}")

    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("="*60)
    print("ContourBasedClustering 参数优化")
    print("="*60)

    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

    print(f"\n数据集: {dataset_name}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")

    # Normalise
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")

    # Parameter search
    best_params, best_labels, best_metrics, all_results = grid_search(
        X, y_true, PARAM_RANGES, OPTIMIZE_METRIC
    )

    if best_params is None:
        print("\n未找到有效的参数组合")
        print("建议:")
        print("  1. 检查数据集是否正确")
        print("  2. 调整参数范围")
        print("  3. 增大density_radius")
    else:
        # Print the best result
        print_best_result("ContourBasedClustering", best_params, best_metrics)

        # Save the figure
        save_path = f"{OUTPUT_DIR}/ContourBased_{dataset_name}.png"
        plot_clustering_result(
            X, best_labels,
            f'ContourBased (ARI={best_metrics["ARI"]:.3f})',
            save_path, dpi=600
        )

        # Save the best result to txt
        txt_path = f"{OUTPUT_DIR}/ContourBased_{dataset_name}_best.txt"
        save_best_result("ContourBasedClustering", dataset_name, best_params, best_metrics, txt_path)

        # Save the ground-truth label figure
        gt_path = f"{OUTPUT_DIR}/GroundTruth_{dataset_name}.png"
        plot_clustering_result(X, y_true, 'Ground Truth', gt_path, dpi=300)
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
