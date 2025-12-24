"""
测试grid_size优化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimize_grid_size import GridSizeOptimizer

# 测试数据集
DATASETS = [
    "data/flame.csv",
    "data/jain.csv",
    "data/Aggregation.csv"
]

def test_single_dataset(dataset_path):
    """测试单个数据集"""
    print("\n" + "=" * 80)
    print(f"测试数据集: {dataset_path}")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv(dataset_path)
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    # 创建优化器
    optimizer = GridSizeOptimizer(X, y_true, max_buffer_distance=5)
    
    print(f"\n初始grid_size: {optimizer.initial_grid_size:.4f}")
    
    # 运行优化
    best_grid_size, best_ari, best_labels = optimizer.optimize(
        initial_step=None,      # 自动设置
        min_step=0.01,
        step_decay=0.5,
        max_iterations=15,
        tolerance=0.001,
        min_grid_size=0.1,
        verbose=True
    )
    
    # 打印历史
    optimizer.print_history()
    
    return optimizer, best_grid_size, best_ari


if __name__ == "__main__":
    print("=" * 80)
    print("grid_size优化测试")
    print("=" * 80)
    
    results = []
    
    # 测试第一个数据集（详细输出）
    optimizer, best_grid_size, best_ari = test_single_dataset(DATASETS[0])
    results.append({
        'dataset': DATASETS[0],
        'optimizer': optimizer,
        'best_grid_size': best_grid_size,
        'best_ari': best_ari
    })
    
    # 绘制优化过程
    print("\n绘制优化过程...")
    optimizer.plot_history()
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("优化结果汇总")
    print("=" * 80)
    print(f"{'数据集':<30} {'初始grid_size':<15} {'最优grid_size':<15} {'初始ARI':<12} {'最优ARI':<12} {'ARI提升':<12}")
    print("-" * 80)
    
    for r in results:
        initial_ari = r['optimizer'].history[0]['ari']
        improvement = r['best_ari'] - initial_ari
        initial_grid = r['optimizer'].initial_grid_size
        
        print(f"{r['dataset']:<30} {initial_grid:<15.4f} {r['best_grid_size']:<15.4f} "
              f"{initial_ari:<12.4f} {r['best_ari']:<12.4f} {improvement:+<12.4f}")
    
    print("=" * 80)
    print("\n测试完成！")
