"""
快速测试optimize_grid_size_by_area.py
测试单个区域的grid_size优化
"""

import json
import numpy as np
import sys
import os
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ContourBasedClusteringWithoutZ22_map as ContourClustering

# ==================== 测试配置 ====================
TARGET_CITY = "长沙市"
TARGET_AREA = "岳麓区"
GRID_SIZE_START = 0.03
GRID_SIZE_END = 0.08
GRID_SIZE_STEP = 0.01
MAX_BUFFER_DISTANCE = 5
MIN_CLUSTER_SIZE = 10
# ==================================================

print("="*80)
print(f"快速测试: {TARGET_CITY} - {TARGET_AREA}")
print("="*80)

# 读取索引文件
with open('client_data_by_city/index.json', 'r', encoding='utf-8') as f:
    index_data = json.load(f)

# 查找目标区域
target_area = None
for area in index_data['city_areas']:
    if area['city'] == TARGET_CITY and area['area'] == TARGET_AREA:
        target_area = area
        break

if not target_area:
    print(f"✗ 未找到区域: {TARGET_CITY} - {TARGET_AREA}")
    exit(1)

print(f"\n区域信息:")
print(f"  城市: {target_area['city']}")
print(f"  区域: {target_area['area']}")
print(f"  客户数: {target_area['count']}")
print(f"  数据文件: {target_area['json_file']}")

# 读取数据
with open(target_area['json_file'], 'r', encoding='utf-8') as f:
    clients = json.load(f)

coords = np.array([[client['lon'], client['lat']] for client in clients])

# 计算密度
tree = cKDTree(coords)
density = np.array([len(tree.query_ball_point(p, 0.01)) for p in coords])

# 准备聚类数据
combined_data = []
for i, client in enumerate(clients):
    combined_data.append((
        client['lon'],
        client['lat'],
        density[i],
        i
    ))

print(f"\n开始扫描grid_size...")
print(f"  范围: {GRID_SIZE_START} - {GRID_SIZE_END}")
print(f"  步长: {GRID_SIZE_STEP}")
print(f"  max_buffer_distance: {MAX_BUFFER_DISTANCE}")
print(f"  min_cluster_size: {MIN_CLUSTER_SIZE}")

# 扫描grid_size
grid_sizes = np.arange(GRID_SIZE_START, GRID_SIZE_END + GRID_SIZE_STEP, GRID_SIZE_STEP)
results = []

for grid_size in grid_sizes:
    try:
        cluster = ContourClustering.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=MAX_BUFFER_DISTANCE,
            min_cluster_size=MIN_CLUSTER_SIZE,
            interactive=False
        )
        
        cluster.perform_clustering()
        labels = cluster.get_predicted_labels()
        labels_array = np.array(labels)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # 计算轮廓系数
        silhouette = None
        if n_clusters > 1 and n_noise < len(labels):
            mask = labels_array != -1
            coords_valid = coords[mask]
            labels_valid = labels_array[mask]
            
            if len(labels_valid) > 0 and len(set(labels_valid)) > 1:
                silhouette = silhouette_score(coords_valid, labels_valid, metric='euclidean')
        
        results.append({
            'grid_size': grid_size,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette
        })
        
        status = f"grid_size={grid_size:.3f}: 簇数={n_clusters}, 噪声={n_noise}"
        if silhouette is not None:
            status += f", 轮廓系数={silhouette:.4f}"
        else:
            status += ", 轮廓系数=N/A"
        print(f"  {status}")
        
    except Exception as e:
        print(f"  grid_size={grid_size:.3f}: 失败 - {e}")

# 找到最佳结果
valid_results = [r for r in results if r['silhouette_score'] is not None]

if valid_results:
    best = max(valid_results, key=lambda x: x['silhouette_score'])
    
    print(f"\n{'='*80}")
    print("✓ 测试成功！")
    print(f"{'='*80}")
    print(f"最佳 grid_size: {best['grid_size']:.3f}")
    print(f"轮廓系数: {best['silhouette_score']:.4f}")
    print(f"簇数: {best['n_clusters']}")
    print(f"噪声点: {best['n_noise']}")
    print(f"{'='*80}")
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    grid_sizes_plot = [r['grid_size'] for r in results]
    silhouette_scores = [r['silhouette_score'] if r['silhouette_score'] is not None else 0 for r in results]
    n_clusters_plot = [r['n_clusters'] for r in results]
    
    ax1.plot(grid_sizes_plot, silhouette_scores, 'b-o', linewidth=2, markersize=6)
    ax1.axvline(best['grid_size'], color='r', linestyle='--', label=f"最佳: {best['grid_size']:.3f}")
    ax1.set_xlabel('grid_size')
    ax1.set_ylabel('轮廓系数')
    ax1.set_title(f'{TARGET_CITY} - {TARGET_AREA}\n轮廓系数 vs grid_size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(grid_sizes_plot, n_clusters_plot, 'g-s', linewidth=2, markersize=6)
    ax2.axvline(best['grid_size'], color='r', linestyle='--', label=f"最佳: {best['grid_size']:.3f}")
    ax2.set_xlabel('grid_size')
    ax2.set_ylabel('簇数')
    ax2.set_title('簇数 vs grid_size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('quick_test_result.png', dpi=100, bbox_inches='tight')
    print(f"\n图表已保存: quick_test_result.png")
    
else:
    print(f"\n{'='*80}")
    print("✗ 测试失败")
    print(f"{'='*80}")
    print("所有grid_size都无法计算轮廓系数")
    print("建议:")
    print("  1. 增大MAX_BUFFER_DISTANCE")
    print("  2. 调整grid_size范围")
    print("  3. 减小MIN_CLUSTER_SIZE")
