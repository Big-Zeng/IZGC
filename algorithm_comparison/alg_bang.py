"""
BANG - 使用pyclustering库的BANG算法
BANG是一种基于网格的多分辨率聚类算法
支持参数循环优化
"""

import numpy as np
import os
from itertools import product

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# 尝试导入pyclustering
try:
    from pyclustering.cluster.bang import bang
    PYCLUSTERING_AVAILABLE = True
except ImportError:
    PYCLUSTERING_AVAILABLE = False
    print("警告: pyclustering未安装，请运行: pip install pyclustering")

# ==================== 配置 ====================
DATASET = "../data/flame.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# 参数搜索范围 (start, end, step)
# pyclustering BANG参数:
# - levels: 网格层级数（决定网格精细程度）
# - density_threshold: 密度阈值（相对值0-1）
PARAM_RANGES = {
    'levels': (4, 100, 1),                  # 网格层级：从3到15，步长1
    'density_threshold': (0.01, 1, 0.05)  # 密度阈值：从0.0到0.3，步长0.05
}

# 优化目标
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        values = list(np.arange(start, end + step/2, step))
        return [round(v, 3) for v in values]


def run_bang(X, levels, density_threshold):
    """
    运行BANG聚类算法 (pyclustering)
    
    参数:
        X: 数据点数组
        levels: 网格层级数
        density_threshold: 密度阈值
    """
    if not PYCLUSTERING_AVAILABLE:
        return None
    
    # 创建BANG实例
    bang_instance = bang(X.tolist(), levels, density_threshold=density_threshold)
    
    # 执行聚类
    bang_instance.process()
    
    # 获取聚类结果
    clusters = bang_instance.get_clusters()
    
    # 转换为标签数组
    labels = np.full(len(X), -1)
    for cluster_id, cluster_points in enumerate(clusters):
        for point_idx in cluster_points:
            labels[point_idx] = cluster_id
    
    return labels


def grid_search(X, y_true, param_ranges, optimize_metric='ARI'):
    """网格搜索最佳参数"""
    if not PYCLUSTERING_AVAILABLE:
        print("pyclustering不可用，跳过BANG")
        return None, None, None, []
    
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
    total_combinations = int(np.prod([len(v) for v in param_values]))
    
    print(f"参数组合总数: {total_combinations}")
    
    results = []
    
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        try:
            labels = run_bang(X, **params)
            
            if labels is None:
                if i % 10 == 0 or i == total_combinations:
                    print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                continue
            
            metrics = evaluate_clustering(y_true, labels, X)
            
            if metrics['n_clusters'] < 2:
                if i % 10 == 0 or i == total_combinations:
                    print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                continue
            
            score = metrics[optimize_metric]
            results.append({'params': params, 'metrics': metrics, 'labels': labels})
            
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = labels.copy()
                best_metrics = metrics
            
            if i % 10 == 0 or i == total_combinations:
                print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
                
        except Exception as e:
            if i % 10 == 0 or i == total_combinations:
                print(f"  进度: {i}/{total_combinations}, 当前最佳{optimize_metric}: {best_score:.4f}")
    
    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("BANG (pyclustering) 参数优化")
    print("="*60)
    
    if not PYCLUSTERING_AVAILABLE:
        print("\npyclustering未安装，请运行: pip install pyclustering")
        exit(1)
    
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
        print_best_result("BANG", best_params, best_metrics)
        
        save_path = f"{OUTPUT_DIR}/BANG_{dataset_name}.png"
        plot_clustering_result(
            X, best_labels,
            f'BANG (ARI={best_metrics["ARI"]:.3f})',
            save_path, dpi=300
        )
        
        # 保存最佳结果到txt
        txt_path = f"{OUTPUT_DIR}/BANG_{dataset_name}_best.txt"
        save_best_result("BANG", dataset_name, best_params, best_metrics, txt_path)
    else:
        print("\n未找到有效的参数组合")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
