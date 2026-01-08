"""
使用ContourBasedClusteringWithoutZ22算法对客户数据进行聚类分析
计算无监督聚类评价指标
"""

import json
import numpy as np
import sys
import os

# 添加父目录到路径以导入ContourBasedClusteringWithoutZ22
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ContourBasedClusteringWithoutZ22 as ContourClustering
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ==================== 配置参数 ====================
GRID_SIZE = 0.05              # 网格大小（经纬度单位）
MAX_BUFFER_DISTANCE = 5       # 最大缓冲距离
MIN_CLUSTER_SIZE = 10         # 最小簇大小（小于此值的簇会被合并）
# ==================================================

print("="*80)
print("客户数据聚类分析")
print("="*80)

# 读取客户数据
print("\n正在读取客户数据...")
with open('client_data.json', 'r', encoding='utf-8') as f:
    clients = json.load(f)

print(f"✓ 加载了 {len(clients)} 个客户")

# 提取坐标数据
coords = np.array([[client['lon'], client['lat']] for client in clients])
print(f"\n数据范围:")
print(f"  经度: [{coords[:, 0].min():.6f}, {coords[:, 0].max():.6f}]")
print(f"  纬度: [{coords[:, 1].min():.6f}, {coords[:, 1].max():.6f}]")

# 计算密度（使用Voronoi图）
print("\n正在计算点密度...")
try:
    vor = Voronoi(coords)
    tree = cKDTree(coords)
    density = np.array([len(tree.query_ball_point(p, 0.1)) for p in coords])
    print(f"✓ 密度计算完成")
    print(f"  密度范围: [{density.min()}, {density.max()}]")
except Exception as e:
    print(f"⚠ Voronoi计算失败，使用简单密度估计: {e}")
    tree = cKDTree(coords)
    density = np.array([len(tree.query_ball_point(p, 0.1)) for p in coords])

# 准备聚类数据（格式：x, y, density, id）
print("\n准备聚类数据...")
combined_data = []
for i, client in enumerate(clients):
    combined_data.append((
        client['lon'],    # x (经度)
        client['lat'],    # y (纬度)
        density[i],       # z (密度)
        i                 # id
    ))

print(f"✓ 数据准备完成，共 {len(combined_data)} 个点")

# 创建聚类实例
print("\n" + "="*80)
print("开始聚类分析")
print("="*80)
print(f"参数设置:")
print(f"  grid_size: {GRID_SIZE}")
print(f"  max_buffer_distance: {MAX_BUFFER_DISTANCE}")
print(f"  min_cluster_size: {MIN_CLUSTER_SIZE}")
print("="*80)

cluster = ContourClustering.ContourBasedClusteringWithoutZ(
    combined_data,
    grid_size=GRID_SIZE,
    max_buffer_distance=MAX_BUFFER_DISTANCE,
    min_cluster_size=MIN_CLUSTER_SIZE
)

# 执行聚类
print("\n正在执行聚类...")
try:
    cluster.perform_clustering()
    print("✓ 聚类完成")
except Exception as e:
    print(f"✗ 聚类失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 获取聚类标签
print("\n正在获取聚类结果...")
pred_labels = cluster.get_predicted_labels()
pred_labels_array = np.array(pred_labels)

# 统计聚类结果
n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
n_noise = list(pred_labels).count(-1)

print("\n" + "="*80)
print("聚类结果统计")
print("="*80)
print(f"发现簇数: {n_clusters}")
print(f"噪声点数: {n_noise}")
print(f"有效聚类点数: {len(pred_labels) - n_noise}")

# 统计每个簇的点数
cluster_counts = {}
for label in pred_labels:
    if label != -1:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

print(f"\n各簇点数分布:")
for label in sorted(cluster_counts.keys()):
    count = cluster_counts[label]
    percentage = (count / len(pred_labels)) * 100
    print(f"  簇 {label}: {count} 个点 ({percentage:.2f}%)")

# 计算无监督评价指标
print("\n" + "="*80)
print("无监督聚类评价指标")
print("="*80)

# 只对非噪声点计算指标
mask = pred_labels_array != -1
coords_valid = coords[mask]
labels_valid = pred_labels_array[mask]

if n_clusters > 1 and len(labels_valid) > 0:
    try:
        # 1. 轮廓系数 (Silhouette Score)
        # 范围: [-1, 1]，越接近1越好
        silhouette = silhouette_score(coords_valid, labels_valid, metric='euclidean')
        print(f"\n1. 轮廓系数 (Silhouette Score): {silhouette:.4f}")
        print(f"   范围: [-1, 1]")
        print(f"   解释: {silhouette:.4f} ", end="")
        if silhouette > 0.7:
            print("(优秀)")
        elif silhouette > 0.5:
            print("(良好)")
        elif silhouette > 0.25:
            print("(一般)")
        else:
            print("(较差)")
        
        # 2. Calinski-Harabasz指数 (方差比准则)
        # 值越大越好
        ch_score = calinski_harabasz_score(coords_valid, labels_valid)
        print(f"\n2. Calinski-Harabasz指数: {ch_score:.4f}")
        print(f"   范围: [0, +∞)")
        print(f"   解释: 值越大，簇间分离度越好")
        
        # 3. Davies-Bouldin指数
        # 值越小越好
        db_score = davies_bouldin_score(coords_valid, labels_valid)
        print(f"\n3. Davies-Bouldin指数: {db_score:.4f}")
        print(f"   范围: [0, +∞)")
        print(f"   解释: {db_score:.4f} ", end="")
        if db_score < 0.5:
            print("(优秀)")
        elif db_score < 1.0:
            print("(良好)")
        elif db_score < 2.0:
            print("(一般)")
        else:
            print("(较差)")
        
        # 4. 簇内误差平方和 (Inertia)
        inertia = 0.0
        for label in np.unique(labels_valid):
            cluster_points = coords_valid[labels_valid == label]
            cluster_center = np.mean(cluster_points, axis=0)
            inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
        
        print(f"\n4. 簇内误差平方和 (Inertia): {inertia:.4f}")
        print(f"   范围: [0, +∞)")
        print(f"   解释: 值越小，簇内紧密度越高")
        
        # 5. 簇间距离统计
        cluster_centers = []
        for label in np.unique(labels_valid):
            cluster_points = coords_valid[labels_valid == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
        
        cluster_centers = np.array(cluster_centers)
        if len(cluster_centers) > 1:
            from scipy.spatial.distance import pdist
            inter_cluster_distances = pdist(cluster_centers)
            
            print(f"\n5. 簇间距离统计:")
            print(f"   最小距离: {inter_cluster_distances.min():.6f}")
            print(f"   最大距离: {inter_cluster_distances.max():.6f}")
            print(f"   平均距离: {inter_cluster_distances.mean():.6f}")
        
        # 保存评价指标
        metrics = {
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'n_valid_points': int(len(labels_valid)),
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(ch_score),
            'davies_bouldin_score': float(db_score),
            'inertia': float(inertia),
            'cluster_counts': {int(k): int(v) for k, v in cluster_counts.items()}
        }
        
        with open('clustering_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 评价指标已保存到: clustering_metrics.json")
        
    except Exception as e:
        print(f"\n✗ 计算评价指标时出错: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n⚠ 无法计算评价指标:")
    if n_clusters <= 1:
        print(f"  - 簇数过少 (n_clusters={n_clusters})")
    if len(labels_valid) == 0:
        print(f"  - 没有有效聚类点")

# 保存聚类结果到客户数据
print("\n正在保存聚类结果...")
for i, client in enumerate(clients):
    client['cluster'] = int(pred_labels[i])

with open('client_data_clustered.json', 'w', encoding='utf-8') as f:
    json.dump(clients, f, ensure_ascii=False, indent=2)

print(f"✓ 聚类结果已保存到: client_data_clustered.json")

# 可视化聚类结果
print("\n正在生成可视化...")
try:
    cluster.visualize_clusters()
    print("✓ 可视化完成")
except Exception as e:
    print(f"⚠ 可视化失败: {e}")

print("\n" + "="*80)
print("聚类分析完成")
print("="*80)
print(f"\n生成的文件:")
print(f"  1. clustering_metrics.json - 评价指标")
print(f"  2. client_data_clustered.json - 带聚类标签的客户数据")
print(f"  3. 可视化图表（如果成功生成）")
print("="*80)
