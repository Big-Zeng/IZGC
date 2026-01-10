"""
CLIQUE 基于网格的聚类算法
支持参数循环优化
"""

import numpy as np
import os
from itertools import product

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# ==================== 配置 ====================
DATASET = "../data/ring.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'grid_size': (1, 50, 0.1),           # 网格大小：从5到25，步长2
    'density_threshold': (1, 25, 1)    # 密度阈值：从2到12，步长2
}

# 优化目标
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_clique(X, grid_size, density_threshold):
    """
    运行CLIQUE聚类算法
    CLIQUE是一种基于网格的子空间聚类算法
    """
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    n_bins_x = max(1, int((x_max - x_min) / grid_size))
    n_bins_y = max(1, int((y_max - y_min) / grid_size))
    
    # 将点分配到网格
    grid_x = np.clip(((X[:, 0] - x_min) / grid_size).astype(int), 0, n_bins_x - 1)
    grid_y = np.clip(((X[:, 1] - y_min) / grid_size).astype(int), 0, n_bins_y - 1)
    
    # 计算每个网格的点
    grid_points = {}
    for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
        key = (gx, gy)
        if key not in grid_points:
            grid_points[key] = []
        grid_points[key].append(i)
    
    # 找出密集网格
    dense_grids = {k: v for k, v in grid_points.items() if len(v) >= density_threshold}
    
    # 连接相邻的密集网格形成簇
    labels = np.full(len(X), -1)
    cluster_id = 0
    visited = set()
    
    for grid_key in dense_grids:
        if grid_key in visited:
            continue
        
        # BFS找到所有连接的密集网格
        queue = [grid_key]
        cluster_grids = []
        
        while queue:
            current = queue.pop(0)
            if current in visited or current not in dense_grids:
                continue
            
            visited.add(current)
            cluster_grids.append(current)
            
            # 检查8个相邻网格
            cx, cy = current
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in dense_grids and neighbor not in visited:
                        queue.append(neighbor)
        
        # 分配标签
        if cluster_grids:
            for gk in cluster_grids:
                for point_idx in dense_grids[gk]:
                    labels[point_idx] = cluster_id
            cluster_id += 1
    
    return labels


def grid_search(X, y_true, param_ranges, optimize_metric='ARI'):
    """网格搜索最佳参数"""
    print("\n开始参数搜索...")
    print(f"优化目标: {optimize_metric}")
    
    # 生成参数值，grid_size保留3位小数
    param_grid = {}
    for name, (start, end, step) in param_ranges.items():
        values = generate_param_values(start, end, step)
        if name == 'grid_size':
            values = [round(v, 3) for v in values]
        param_grid[name] = values
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
            labels = run_clique(X, **params)
            metrics = evaluate_clustering(y_true, labels, X)
            
            if metrics['n_clusters'] < 2:
                if i % 20 == 0 or i == total_combinations:
                    print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                continue
            
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
            if i % 20 == 0 or i == total_combinations:
                print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
    
    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("CLIQUE 参数优化")
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
        print_best_result("CLIQUE", best_params, best_metrics)
        
        save_path = f"{OUTPUT_DIR}/CLIQUE_{dataset_name}.png"
        plot_clustering_result(
            X, best_labels,
            f'CLIQUE (ARI={best_metrics["ARI"]:.3f})',
            save_path, dpi=300
        )
        
        # 保存最佳结果到txt
        txt_path = f"{OUTPUT_DIR}/CLIQUE_{dataset_name}_best.txt"
        save_best_result("CLIQUE", dataset_name, best_params, best_metrics, txt_path)
    else:
        print("\n未找到有效的参数组合")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
