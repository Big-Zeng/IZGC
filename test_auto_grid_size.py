"""
测试自动计算grid_size功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 简单的测试函数
def calculate_auto_grid_size(coords, grid_multiplier=2.0):
    """
    自动计算grid_size
    
    参数:
        coords: 坐标数组 shape=(n, 2)
        grid_multiplier: 格子数量倍数
        
    返回:
        grid_size: 计算得到的网格大小
    """
    n_points = len(coords)
    
    # 计算包裹所有点的矩形
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # 矩形面积
    area = width * height
    
    # 目标格子数量
    target_grid_count = n_points * grid_multiplier
    
    # 计算grid_size
    grid_size = np.sqrt(area / target_grid_count)
    
    # 实际格子数
    actual_grid_count = int(width / grid_size) * int(height / grid_size)
    
    return {
        'grid_size': grid_size,
        'n_points': n_points,
        'width': width,
        'height': height,
        'area': area,
        'target_grids': target_grid_count,
        'actual_grids': actual_grid_count,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }


def visualize_grid(coords, grid_size, title="Grid Visualization"):
    """可视化网格"""
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # 扩展边界
    x_start = x_min - 2 * grid_size
    x_end = x_max + 2 * grid_size
    y_start = y_min - 2 * grid_size
    y_end = y_max + 2 * grid_size
    
    # 创建网格线
    x_ticks = np.arange(x_start, x_end + grid_size, grid_size)
    y_ticks = np.arange(y_start, y_end + grid_size, grid_size)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制网格线
    for x in x_ticks:
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    for y in y_ticks:
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 绘制点
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=5, label='数据点')
    
    # 绘制边界矩形
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             'b-', linewidth=2, label='数据边界')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.axis('equal')
    plt.tight_layout()


def test_dataset(dataset_name, dataset_file, multipliers=[1.0, 2.0, 3.0]):
    """测试单个数据集"""
    print("=" * 80)
    print(f"测试数据集: {dataset_name}")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv(dataset_file)
    coords = df[['x', 'y']].values
    
    print(f"\n数据集信息:")
    print(f"  点数: {len(coords)}")
    print(f"  X范围: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  Y范围: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    
    # 测试不同的multiplier
    results = []
    for multiplier in multipliers:
        result = calculate_auto_grid_size(coords, multiplier)
        results.append(result)
        
        print(f"\n倍数 = {multiplier}:")
        print(f"  grid_size: {result['grid_size']:.4f}")
        print(f"  目标格子数: {result['target_grids']:.0f}")
        print(f"  实际格子数: {result['actual_grids']}")
        print(f"  误差: {abs(result['actual_grids'] - result['target_grids']):.0f} "
              f"({abs(result['actual_grids'] - result['target_grids']) / result['target_grids'] * 100:.1f}%)")
    
    # 可视化推荐的grid_size（multiplier=2.0）
    recommended = results[1] if len(results) > 1 else results[0]
    visualize_grid(coords, recommended['grid_size'],
                  f"{dataset_name} - grid_size={recommended['grid_size']:.4f} (倍数=2.0)")
    
    return results


def compare_multipliers():
    """对比不同multiplier的效果"""
    datasets = [
        ('flame', 'data/flame.csv'),
        ('Aggregation', 'data/Aggregation.csv'),
        ('jain', 'data/jain.csv'),
    ]
    
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print("\n" + "=" * 100)
    print("不同倍数的grid_size对比")
    print("=" * 100)
    print(f"{'数据集':<15} {'点数':<8} {'倍数':<8} {'grid_size':<12} {'目标格子':<12} {'实际格子':<12} {'误差%':<10}")
    print("-" * 100)
    
    for dataset_name, dataset_file in datasets:
        try:
            df = pd.read_csv(dataset_file)
            coords = df[['x', 'y']].values
            
            for multiplier in multipliers:
                result = calculate_auto_grid_size(coords, multiplier)
                error_pct = abs(result['actual_grids'] - result['target_grids']) / result['target_grids'] * 100
                
                print(f"{dataset_name:<15} {result['n_points']:<8} {multiplier:<8.1f} "
                      f"{result['grid_size']:<12.4f} {result['target_grids']:<12.0f} "
                      f"{result['actual_grids']:<12} {error_pct:<10.1f}")
        except FileNotFoundError:
            print(f"{dataset_name:<15} 文件未找到")
    
    print("=" * 100)


if __name__ == "__main__":
    print("自动计算grid_size功能测试\n")
    
    # 测试1: 单个数据集详细测试
    try:
        test_dataset('jain', 'data/jain.csv', multipliers=[1.0, 2.0, 3.0])
        plt.show()
    except FileNotFoundError:
        print("错误: data/flame.csv 未找到")
    
    # 测试2: 对比不同数据集和倍数
    print("\n")
    compare_multipliers()
    
    # 测试3: 推荐配置
    print("\n" + "=" * 80)
    print("推荐配置")
    print("=" * 80)
    print("""
对于大多数数据集，推荐使用:

CONTOUR_PARAMS = {
    'grid_size': None,               # 自动计算
    'max_buffer_distance': 5,
    'auto_grid_multiplier': 2.0      # 格子数 = 点数 × 2
}

这样可以:
✅ 自动适应不同规模的数据集
✅ 平衡计算性能和聚类精度
✅ 减少手动调参的工作量

如果需要更精细的聚类，可以增加到 3.0
如果数据集很大（>2000点），可以减少到 1.5 或 1.0
    """)
