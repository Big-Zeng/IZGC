"""
按城市优化grid_size参数
对每个市的客户数据进行grid_size扫描，找到使轮廓系数最佳的grid_size值
"""

import json
import numpy as np
import sys
import os
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互模式

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ContourBasedClusteringWithoutZ22_map as ContourClustering

# ==================== 坐标转换函数 ====================
def lonlat_to_xy(lon, lat, lon_ref, lat_ref):
    """
    将经纬度转换为平面坐标（单位：公里）
    使用简化的墨卡托投影
    
    参数:
        lon, lat: 要转换的经纬度
        lon_ref, lat_ref: 参考点经纬度（通常是区域中心）
    
    返回:
        x, y: 平面坐标（公里）
    """
    # 地球半径（公里）
    R = 6371.0
    
    # 转换为弧度
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    lon_ref_rad = np.radians(lon_ref)
    lat_ref_rad = np.radians(lat_ref)
    
    # 计算平面坐标（公里）
    x = R * (lon_rad - lon_ref_rad) * np.cos(lat_ref_rad)
    y = R * (lat_rad - lat_ref_rad)
    
    return x, y

def normalize_coordinates(coords):
    """
    归一化坐标到[0, 100]范围
    
    参数:
        coords: 坐标数组 (n_samples, 2)
        
    返回:
        coords_normalized: 归一化后的坐标
        scale_info: 缩放信息字典
    """
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 归一化到[0, 100]
    coords_normalized = np.zeros_like(coords)
    coords_normalized[:, 0] = ((coords[:, 0] - x_min) / x_range * 100) if x_range > 0 else 50
    coords_normalized[:, 1] = ((coords[:, 1] - y_min) / y_range * 100) if y_range > 0 else 50
    
    scale_info = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'x_range': x_range,
        'y_range': y_range
    }
    
    return coords_normalized, scale_info
# ======================================================

# ==================== 配置参数 ====================
# 城市选择（留空则处理所有城市）
TARGET_CITY = "长沙市"            # 指定城市名称，如 "长沙市"，留空处理所有城市

GRID_SIZE_START = 6      # grid_size起始值
GRID_SIZE_END = 7         # grid_size结束值
GRID_SIZE_STEP = 0.001       # grid_size步长
MAX_BUFFER_DISTANCE = 0     # 最大缓冲距离
MIN_CLUSTER_SIZE = 5       # 最小簇大小
MIN_POINTS = 50             # 最少点数（少于此值的城市跳过）
# ==================================================

print("="*80)
print("按城市优化grid_size参数（基于轮廓系数）")
print("="*80)
print(f"\n参数设置:")
if TARGET_CITY:
    print(f"  目标城市: {TARGET_CITY}")
else:
    print(f"  目标: 所有城市")
print(f"  grid_size范围: {GRID_SIZE_START} - {GRID_SIZE_END}")
print(f"  步长: {GRID_SIZE_STEP}")
print(f"  max_buffer_distance: {MAX_BUFFER_DISTANCE}")
print(f"  min_cluster_size: {MIN_CLUSTER_SIZE}")
print(f"  最少点数: {MIN_POINTS}")
print(f"  评价指标: 轮廓系数 (Silhouette Score)")
print("="*80)

# 读取索引文件
print("\n正在读取索引文件...")
with open('client_data_by_city/index.json', 'r', encoding='utf-8') as f:
    index_data = json.load(f)

# 筛选符合条件的城市（点数 >= MIN_POINTS）
valid_cities = [city for city in index_data['cities'] if city['count'] >= MIN_POINTS and city['city'] is not None]

# 根据TARGET_CITY进一步筛选
if TARGET_CITY:
    valid_cities = [city for city in valid_cities if city['city'] == TARGET_CITY]
    print(f"✓ 筛选城市: {TARGET_CITY}")

print(f"✓ 找到 {len(valid_cities)} 个符合条件的城市（点数 >= {MIN_POINTS}）")

if len(valid_cities) == 0:
    print("\n✗ 没有符合条件的城市！")
    print("建议:")
    print("  1. 检查TARGET_CITY是否正确")
    print("  2. 降低MIN_POINTS阈值")
    print("  3. 查看 client_data_by_city/index.json 了解可用的城市")
    exit(1)

# 创建输出目录
output_dir = 'grid_size_optimization_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ 创建输出目录: {output_dir}")

# 存储所有结果
all_results = []

# 对每个城市进行grid_size扫描
for idx, city_info in enumerate(valid_cities, 1):
    city = city_info['city']
    count = city_info['count']
    json_file = city_info['json_file']
    
    print(f"\n{'='*80}")
    print(f"[{idx}/{len(valid_cities)}] 处理: {city} ({count}个客户)")
    print(f"{'='*80}")
    
    # 读取城市数据
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            clients = json.load(f)
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        continue
    
    # 提取经纬度坐标
    lonlat_coords = np.array([[client['lon'], client['lat']] for client in clients])
    
    # 计算参考点（区域中心）
    lon_ref = np.mean(lonlat_coords[:, 0])
    lat_ref = np.mean(lonlat_coords[:, 1])
    
    print(f"  参考点（区域中心）: 经度={lon_ref:.6f}, 纬度={lat_ref:.6f}")
    
    # 将经纬度转换为平面坐标（公里）
    xy_coords = np.array([lonlat_to_xy(lon, lat, lon_ref, lat_ref) 
                          for lon, lat in lonlat_coords])
    
    print(f"  平面坐标范围: X=[{xy_coords[:, 0].min():.3f}, {xy_coords[:, 0].max():.3f}] km")
    print(f"                Y=[{xy_coords[:, 1].min():.3f}, {xy_coords[:, 1].max():.3f}] km")
    
    # 归一化坐标到[0, 100]
    coords, scale_info = normalize_coordinates(xy_coords)
    
    print(f"  归一化后范围: X=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"                Y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    
    # 计算密度（使用归一化后的坐标）
    tree = cKDTree(coords)
    density = np.array([len(tree.query_ball_point(p, 5)) for p in coords])
    
    # 准备聚类数据（使用归一化后的坐标）
    combined_data = []
    for i in range(len(coords)):
        combined_data.append((
            coords[i, 0],
            coords[i, 1],
            density[i],
            i
        ))
    
    # 扫描grid_size
    grid_sizes = np.arange(GRID_SIZE_START, GRID_SIZE_END + GRID_SIZE_STEP, GRID_SIZE_STEP)
    results = []
    
    print(f"\n开始扫描grid_size ({len(grid_sizes)}个值)...")
    
    for grid_size in grid_sizes:
        try:
            # 创建聚类实例
            cluster = ContourClustering.ContourBasedClusteringWithoutZ(
                combined_data,
                grid_size=grid_size,
                max_buffer_distance=MAX_BUFFER_DISTANCE,
                min_cluster_size=MIN_CLUSTER_SIZE,
                interactive=False
            )
            
            # 执行聚类
            cluster.perform_clustering()
            
            # 获取聚类标签
            labels = cluster.get_predicted_labels()
            labels_array = np.array(labels)
            
            # 统计簇数
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
            results.append({
                'grid_size': grid_size,
                'n_clusters': 0,
                'n_noise': 0,
                'silhouette_score': None
            })
    
    # 找到最佳grid_size
    valid_results = [r for r in results if r['silhouette_score'] is not None]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['silhouette_score'])
        
        print(f"\n✓ 最佳结果:")
        print(f"  grid_size: {best_result['grid_size']:.3f}")
        print(f"  轮廓系数: {best_result['silhouette_score']:.4f}")
        print(f"  簇数: {best_result['n_clusters']}")
        print(f"  噪声点: {best_result['n_noise']}")
        
        # 保存结果
        city_result = {
            'city': city,
            'count': count,
            'best_grid_size': best_result['grid_size'],
            'best_silhouette_score': best_result['silhouette_score'],
            'best_n_clusters': best_result['n_clusters'],
            'best_n_noise': best_result['n_noise'],
            'all_results': results
        }
        all_results.append(city_result)
        
        # 绘制grid_size vs 轮廓系数曲线
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 轮廓系数曲线
            grid_sizes_plot = [r['grid_size'] for r in results]
            silhouette_scores = [r['silhouette_score'] if r['silhouette_score'] is not None else 0 for r in results]
            
            ax1.plot(grid_sizes_plot, silhouette_scores, 'b-o', linewidth=2, markersize=4)
            ax1.axvline(best_result['grid_size'], color='r', linestyle='--', label=f"最佳: {best_result['grid_size']:.3f}")
            ax1.set_xlabel('grid_size')
            ax1.set_ylabel('轮廓系数 (Silhouette Score)')
            ax1.set_title(f'{city}\n轮廓系数 vs grid_size')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 簇数曲线
            n_clusters_plot = [r['n_clusters'] for r in results]
            ax2.plot(grid_sizes_plot, n_clusters_plot, 'g-s', linewidth=2, markersize=4)
            ax2.axvline(best_result['grid_size'], color='r', linestyle='--', label=f"最佳: {best_result['grid_size']:.3f}")
            ax2.set_xlabel('grid_size')
            ax2.set_ylabel('簇数')
            ax2.set_title('簇数 vs grid_size')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # 保存图片
            safe_city = city.replace('/', '_').replace('\\', '_') if city else '未知城市'
            plot_file = f"{output_dir}/{safe_city}.png"
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  图表已保存: {plot_file}")
        except Exception as e:
            print(f"  ⚠ 绘图失败: {e}")
    else:
        print(f"\n✗ 未找到有效结果（所有grid_size都无法计算轮廓系数）")

# 保存所有结果
print(f"\n{'='*80}")
print("保存结果")
print(f"{'='*80}")

# 保存JSON格式
json_file = f"{output_dir}/optimization_results.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"✓ JSON结果已保存: {json_file}")

# 生成汇总报告
report_lines = []
report_lines.append("# grid_size优化结果汇总报告\n\n")
report_lines.append(f"## 参数设置\n\n")
report_lines.append(f"- grid_size范围: {GRID_SIZE_START} - {GRID_SIZE_END}\n")
report_lines.append(f"- 步长: {GRID_SIZE_STEP}\n")
report_lines.append(f"- max_buffer_distance: {MAX_BUFFER_DISTANCE}\n")
report_lines.append(f"- min_cluster_size: {MIN_CLUSTER_SIZE}\n")
report_lines.append(f"- 最少点数: {MIN_POINTS}\n\n")

report_lines.append(f"## 优化结果\n\n")
report_lines.append(f"成功优化的城市数: {len(all_results)}\n\n")

report_lines.append("| 城市 | 客户数 | 最佳grid_size | 轮廓系数 | 簇数 | 噪声点 |\n")
report_lines.append("|------|--------|---------------|----------|------|--------|\n")

for result in sorted(all_results, key=lambda x: x['best_silhouette_score'], reverse=True):
    report_lines.append(f"| {result['city']} | {result['count']} | "
                       f"{result['best_grid_size']:.3f} | {result['best_silhouette_score']:.4f} | "
                       f"{result['best_n_clusters']} | {result['best_n_noise']} |\n")

# 统计分析
if all_results:
    best_grid_sizes = [r['best_grid_size'] for r in all_results]
    silhouette_scores = [r['best_silhouette_score'] for r in all_results]
    
    report_lines.append(f"\n## 统计分析\n\n")
    report_lines.append(f"- 最佳grid_size平均值: {np.mean(best_grid_sizes):.3f}\n")
    report_lines.append(f"- 最佳grid_size中位数: {np.median(best_grid_sizes):.3f}\n")
    report_lines.append(f"- 最佳grid_size标准差: {np.std(best_grid_sizes):.3f}\n")
    report_lines.append(f"- 最佳grid_size范围: [{min(best_grid_sizes):.3f}, {max(best_grid_sizes):.3f}]\n\n")
    
    report_lines.append(f"- 轮廓系数平均值: {np.mean(silhouette_scores):.4f}\n")
    report_lines.append(f"- 轮廓系数中位数: {np.median(silhouette_scores):.4f}\n")
    report_lines.append(f"- 轮廓系数范围: [{min(silhouette_scores):.4f}, {max(silhouette_scores):.4f}]\n\n")
    
    # 推荐grid_size
    recommended_grid_size = np.median(best_grid_sizes)
    report_lines.append(f"## 推荐参数\n\n")
    report_lines.append(f"基于所有城市的优化结果，推荐使用:\n\n")
    report_lines.append(f"- **grid_size**: {recommended_grid_size:.3f}\n")
    report_lines.append(f"- max_buffer_distance: {MAX_BUFFER_DISTANCE}\n")
    report_lines.append(f"- min_cluster_size: {MIN_CLUSTER_SIZE}\n\n")
    report_lines.append(f"- **grid_size**: {recommended_grid_size:.3f}\n")
    report_lines.append(f"- max_buffer_distance: {MAX_BUFFER_DISTANCE}\n")
    report_lines.append(f"- min_cluster_size: {MIN_CLUSTER_SIZE}\n\n")

report_file = f"{output_dir}/optimization_summary.md"
with open(report_file, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)
print(f"✓ 汇总报告已保存: {report_file}")

# 绘制汇总图表
if all_results:
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 最佳grid_size分布
        best_grid_sizes = [r['best_grid_size'] for r in all_results]
        ax1.hist(best_grid_sizes, bins=20, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(best_grid_sizes), color='r', linestyle='--', label=f'平均值: {np.mean(best_grid_sizes):.3f}')
        ax1.axvline(np.median(best_grid_sizes), color='g', linestyle='--', label=f'中位数: {np.median(best_grid_sizes):.3f}')
        ax1.set_xlabel('最佳grid_size')
        ax1.set_ylabel('城市数量')
        ax1.set_title('最佳grid_size分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 轮廓系数分布
        silhouette_scores = [r['best_silhouette_score'] for r in all_results]
        ax2.hist(silhouette_scores, bins=20, color='lightgreen', edgecolor='black')
        ax2.axvline(np.mean(silhouette_scores), color='r', linestyle='--', label=f'平均值: {np.mean(silhouette_scores):.4f}')
        ax2.set_xlabel('轮廓系数')
        ax2.set_ylabel('城市数量')
        ax2.set_title('轮廓系数分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. grid_size vs 轮廓系数散点图
        ax3.scatter(best_grid_sizes, silhouette_scores, alpha=0.6, s=50)
        ax3.set_xlabel('最佳grid_size')
        ax3.set_ylabel('轮廓系数')
        ax3.set_title('grid_size vs 轮廓系数')
        ax3.grid(True, alpha=0.3)
        
        # 4. 客户数 vs 最佳grid_size
        counts = [r['count'] for r in all_results]
        scatter = ax4.scatter(counts, best_grid_sizes, alpha=0.6, s=100, c=silhouette_scores, cmap='viridis')
        ax4.set_xlabel('客户数')
        ax4.set_ylabel('最佳grid_size')
        ax4.set_title('客户数 vs 最佳grid_size（颜色=轮廓系数）')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('轮廓系数')
        
        plt.tight_layout()
        summary_plot = f"{output_dir}/optimization_summary.png"
        plt.savefig(summary_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 汇总图表已保存: {summary_plot}")
    except Exception as e:
        print(f"⚠ 绘制汇总图表失败: {e}")

# 打印摘要
print(f"\n{'='*80}")
print("优化完成摘要")
print(f"{'='*80}")
print(f"\n处理的城市数: {len(all_results)}")

if all_results:
    print(f"\n最佳grid_size统计:")
    print(f"  平均值: {np.mean(best_grid_sizes):.3f}")
    print(f"  中位数: {np.median(best_grid_sizes):.3f}")
    print(f"  标准差: {np.std(best_grid_sizes):.3f}")
    print(f"  范围: [{min(best_grid_sizes):.3f}, {max(best_grid_sizes):.3f}]")
    
    print(f"\n轮廓系数统计:")
    print(f"  平均值: {np.mean(silhouette_scores):.4f}")
    print(f"  中位数: {np.median(silhouette_scores):.4f}")
    print(f"  范围: [{min(silhouette_scores):.4f}, {max(silhouette_scores):.4f}]")
    
    print(f"\n推荐grid_size: {np.median(best_grid_sizes):.3f}")
    
    print(f"\n前5个最佳结果:")
    for i, result in enumerate(sorted(all_results, key=lambda x: x['best_silhouette_score'], reverse=True)[:5], 1):
        print(f"  {i}. {result['city']}: "
              f"grid_size={result['best_grid_size']:.3f}, "
              f"轮廓系数={result['best_silhouette_score']:.4f}")

print(f"\n输出目录: {output_dir}/")
print(f"  - JSON结果: {json_file}")
print(f"  - 汇总报告: {report_file}")
print(f"  - 各城市图表: {len(all_results)} 个PNG文件")

print(f"\n{'='*80}")
print("✓ 优化完成")
print(f"{'='*80}")
