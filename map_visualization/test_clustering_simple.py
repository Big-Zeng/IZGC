"""
简单测试聚类功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ContourBasedClusteringWithoutZ22_map as ContourClustering
import numpy as np

# 创建简单测试数据
test_points = [
    (112.9, 28.2, 10, 1),
    (112.91, 28.21, 12, 2),
    (112.92, 28.22, 8, 3),
    (113.0, 28.3, 15, 4),
    (113.01, 28.31, 11, 5),
]

print("创建聚类实例...")
cluster = ContourClustering.ContourBasedClusteringWithoutZ(
    test_points,
    grid_size=0.05,
    max_buffer_distance=5,
    min_cluster_size=2,
    interactive=False  # 非交互模式
)

print("执行聚类...")
cluster.perform_clustering()

print("获取聚类标签...")
labels = cluster.get_predicted_labels()
print(f"标签: {labels}")

print("获取密度中心...")
print(f"密度中心数量: {len(cluster.density_centers)}")

# 测试获取多边形数据
print("\n测试获取多边形数据:")
for i, center in enumerate(cluster.density_centers):
    print(f"簇 {i}:")
    print(f"  多边形点数: {len(center.polygon)}")
    print(f"  中心: {center.center}")
    print(f"  多边形前3个点: {center.polygon[:3].tolist() if len(center.polygon) > 0 else []}")

print("\n✓ 测试完成")
