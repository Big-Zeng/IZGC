"""
Grid Size诊断工具
帮助理解为什么某些grid_size值导致ARI为0或聚类失败
"""

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

import ContourBasedClusteringWithoutZ4 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics


def diagnose_grid_size(dataset_path, grid_size, max_buffer_distance=5):
    """诊断单个grid_size值"""
    print("="*80)
    print(f"诊断 grid_size = {grid_size}")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv(dataset_path)
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    print(f"\n数据集信息:")
    print(f"  样本数: {len(X)}")
    print(f"  真实簇数: {len(np.unique(y_true))}")
    print(f"  X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # 计算数据范围
    x_range = X[:, 0].max() - X[:, 0].min()
    y_range = X[:, 1].max() - X[:, 1].min()
    area = x_range * y_range
    
    print(f"\n数据范围:")
    print(f"  X范围: {x_range:.2f}")
    print(f"  Y范围: {y_range:.2f}")
    print(f"  总面积: {area:.2f}")
    
    # 估算网格数量
    n_grids_x = int(x_range / grid_size) + 4  # +4 for padding
    n_grids_y = int(y_range / grid_size) + 4
    total_grids = n_grids_x * n_grids_y
    
    print(f"\n网格信息:")
    print(f"  grid_size: {grid_size}")
    print(f"  X方向网格数: {n_grids_x}")
    print(f"  Y方向网格数: {n_grids_y}")
    print(f"  总网格数: {total_grids}")
    print(f"  点数/网格数比: {len(X) / total_grids:.2f}")
    
    # 计算密度
    vor = Voronoi(X)
    density = np.zeros(len(X))
    
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            density[i] = 0
            continue
        
        polygon = vor.vertices[region]
        x = polygon[:, 0]
        y = polygon[:, 1]
        area_poly = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        density[i] = 1.0 / area_poly if area_poly > 0 else 0
    
    print(f"\n密度信息:")
    print(f"  密度范围: [{density.min():.4f}, {density.max():.4f}]")
    print(f"  平均密度: {density.mean():.4f}")
    print(f"  密度标准差: {density.std():.4f}")
    
    # 准备数据
    combined_data = [(*point, density[i], i) for i, point in enumerate(X)]
    
    # 创建聚类实例
    try:
        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance,
            interpolation_method='linear'
        )
        
        # 执行聚类（禁用可视化）
        import matplotlib
        matplotlib.use('Agg')
        
        # 简化虚拟点生成
        def silent_generate():
            x_coords = [p[0] for p in cluster.actual_points]
            y_coords = [p[1] for p in cluster.actual_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            i_start = min_x - 2 * cluster.grid_size
            i_end = max_x + 2 * cluster.grid_size
            j_start = min_y - 2 * cluster.grid_size
            j_end = max_y + 2 * cluster.grid_size
            
            virtual_points = []
            i = i_start
            while i < i_end:
                j = j_start
                while j < j_end:
                    has_point = any(
                        i <= x < i + cluster.grid_size and
                        j <= y < j + cluster.grid_size
                        for (x, y, _, _) in cluster.actual_points
                    )
                    if not has_point:
                        virtual_points.append((i + cluster.grid_size / 2,
                                             j + cluster.grid_size / 2,
                                             0.0, -1))
                    j += cluster.grid_size
                i += cluster.grid_size
            
            cluster.virtual_points = virtual_points
            
            x_ticks = np.arange(i_start, i_end + cluster.grid_size, cluster.grid_size)
            y_ticks = np.arange(j_start, j_end + cluster.grid_size, cluster.grid_size)
            
            return virtual_points, x_ticks, y_ticks
        
        cluster.generate_virtual_points = silent_generate
        
        # 执行聚类
        cluster.perform_clustering()
        
        # 获取结果
        pred_labels = cluster.get_predicted_labels()
        
        # 分析结果
        unique_labels = set(pred_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(pred_labels).count(-1)
        
        print(f"\n聚类结果:")
        print(f"  发现簇数: {n_clusters}")
        print(f"  噪声点数: {n_noise}")
        print(f"  有效点数: {len(pred_labels) - n_noise}")
        
        # 每个簇的点数
        if n_clusters > 0:
            print(f"\n每个簇的点数:")
            for label in sorted(unique_labels):
                if label == -1:
                    continue
                count = list(pred_labels).count(label)
                percentage = count / len(pred_labels) * 100
                print(f"    簇 {label}: {count} 点 ({percentage:.1f}%)")
        
        # 计算评估指标
        if n_clusters > 0:
            metrics = ClusteringMetrics.evaluate_all(y_true, pred_labels)
            
            print(f"\n评估指标:")
            print(f"  ARI: {metrics['ARI']:.4f}")
            print(f"  AMI: {metrics['AMI']:.4f}")
            print(f"  FMI: {metrics['FMI']:.4f}")
            
            # 分析ARI为0的原因
            if abs(metrics['ARI']) < 0.001:
                print(f"\n⚠️  ARI接近0的可能原因:")
                if n_clusters == 1:
                    print(f"    - 只产生了1个簇，没有分离")
                    print(f"    - 建议: 减小grid_size或增大max_buffer_distance")
                elif n_noise > len(pred_labels) * 0.5:
                    print(f"    - 噪声点过多 ({n_noise}/{len(pred_labels)})")
                    print(f"    - 建议: 增大max_buffer_distance")
                else:
                    print(f"    - 聚类结果与真实标签完全不匹配")
                    print(f"    - 建议: 调整grid_size或检查数据")
        else:
            print(f"\n❌ 聚类失败:")
            print(f"    - 没有产生任何簇")
            print(f"    - 所有点都被标记为噪声")
            print(f"    - 建议: 大幅增大max_buffer_distance或调整grid_size")
        
        # 可视化建议
        print(f"\n建议:")
        if grid_size < 0.1:
            print(f"  - grid_size太小 ({grid_size:.3f})，网格过密")
            print(f"  - 建议增大到 0.1-1.0 范围")
        elif grid_size > x_range / 2:
            print(f"  - grid_size太大 ({grid_size:.3f})，网格过疏")
            print(f"  - 建议减小到 {x_range/10:.2f}-{x_range/5:.2f} 范围")
        else:
            print(f"  - grid_size在合理范围内")
        
        if total_grids > 10000:
            print(f"  - 网格数量过多 ({total_grids})，计算量大")
            print(f"  - 建议增大grid_size")
        elif total_grids < 50:
            print(f"  - 网格数量过少 ({total_grids})，分辨率低")
            print(f"  - 建议减小grid_size")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 执行失败:")
        print(f"  错误: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    
    # 默认参数
    dataset = "data/flame.csv"
    
    print("="*80)
    print("Grid Size诊断工具")
    print("="*80)
    
    # 测试几个典型的grid_size值
    test_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\n将测试以下grid_size值: {test_values}")
    print(f"数据集: {dataset}\n")
    
    for grid_size in test_values:
        diagnose_grid_size(dataset, grid_size, max_buffer_distance=5)
        print("\n" + "-"*80 + "\n")
    
    print("="*80)
    print("诊断完成")
    print("="*80)
    print("\n根据诊断结果:")
    print("  1. 选择产生2个簇且ARI较高的grid_size")
    print("  2. 避免产生1个簇或全是噪声的grid_size")
    print("  3. 在合理范围内调整grid_size和max_buffer_distance")
