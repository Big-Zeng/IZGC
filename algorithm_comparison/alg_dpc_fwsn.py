"""
DPC-FWSN 密度峰值聚类算法 (MATLAB实现)
支持参数循环优化
"""

import numpy as np
import os
import sys
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, print_best_result, save_best_result
)

# 尝试导入MATLAB集成
try:
    from matlab_integration import MatlabRunner
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("警告: matlab_integration模块未找到，MATLAB算法将不可用")

# ==================== 配置 ====================
DATASET = "../data/2d_3c.csv"
OUTPUT_DIR = "results"
NORMALIZE = True

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'k': (65, 66, 1)  # 近邻数量: 从5到20，步长2
}

# 优化目标: 'ARI', 'AMI', 'FMI', 'Silhouette'
OPTIMIZE_METRIC = 'ARI'
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_dpc_fwsn(X, y_true, k, matlab_runner=None):
    """运行DPC-FWSN聚类"""
    if not MATLAB_AVAILABLE:
        return None
    
    n_clusters = len(np.unique(y_true))
    data_with_labels = np.column_stack([X, y_true])
    
    if matlab_runner is None:
        matlab_runner = MatlabRunner(method='auto')
    
    labels, metrics = matlab_runner.run_algorithm_pr(
        data_with_labels, n_clusters, k
    )
    
    return labels, matlab_runner


def grid_search(X, y_true, param_ranges, optimize_metric='ARI'):
    """网格搜索最佳参数"""
    if not MATLAB_AVAILABLE:
        print("MATLAB不可用，跳过DPC-FWSN")
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
    total_combinations = np.prod([len(v) for v in param_values])
    
    print(f"参数组合数: {total_combinations}")
    
    results = []
    matlab_runner = None
    
    try:
        for i, values in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, values))
            
            try:
                labels, matlab_runner = run_dpc_fwsn(X, y_true, **params, matlab_runner=matlab_runner)
                
                if labels is None:
                    continue
                
                metrics = evaluate_clustering(y_true, labels, X)
                
                if metrics['n_clusters'] < 2:
                    continue
                
                score = metrics[optimize_metric]
                results.append({'params': params, 'metrics': metrics, 'labels': labels})
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_labels = labels.copy()
                    best_metrics = metrics.copy()
                
                print(f"  k={params['k']}: {optimize_metric}={score:.4f}")
                    
            except Exception as e:
                print(f"  参数 {params} 失败: {e}")
    finally:
        if matlab_runner:
            matlab_runner.close()
    
    return best_params, best_labels, best_metrics, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("DPC-FWSN (MATLAB) 参数优化")
    print("="*60)
    
    if not MATLAB_AVAILABLE:
        print("\nMATLAB不可用，无法运行DPC-FWSN算法")
        print("请确保:")
        print("  1. 已安装MATLAB")
        print("  2. 已安装MATLAB Engine for Python")
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
        print_best_result("DPC-FWSN", best_params, best_metrics)
        
        save_path = f"{OUTPUT_DIR}/DPC-FWSN_{dataset_name}.png"
        plot_clustering_result(
            X, best_labels,
            f'DPC-FWSN (ARI={best_metrics["ARI"]:.3f})',
            save_path, dpi=300
        )
        
        # 保存最佳结果到txt
        txt_path = f"{OUTPUT_DIR}/DPC-FWSN_{dataset_name}_best.txt"
        save_best_result("DPC-FWSN", dataset_name, best_params, best_metrics, txt_path)
    else:
        print("\n未找到有效的参数组合")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
