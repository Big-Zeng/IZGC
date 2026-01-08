"""测试经纬度到平面坐标的转换和归一化"""
import json
import numpy as np

def lonlat_to_xy(lon, lat, lon_ref, lat_ref):
    """将经纬度转换为平面坐标（单位：公里）"""
    R = 6371.0  # 地球半径（公里）
    
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    lon_ref_rad = np.radians(lon_ref)
    lat_ref_rad = np.radians(lat_ref)
    
    x = R * (lon_rad - lon_ref_rad) * np.cos(lat_ref_rad)
    y = R * (lat_rad - lat_ref_rad)
    
    return x, y

def normalize_coordinates(coords):
    """归一化坐标到[0, 100]范围"""
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    coords_normalized = np.zeros_like(coords)
    coords_normalized[:, 0] = ((coords[:, 0] - x_min) / x_range * 100) if x_range > 0 else 50
    coords_normalized[:, 1] = ((coords[:, 1] - y_min) / y_range * 100) if y_range > 0 else 50
    
    scale_info = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'x_range': x_range, 'y_range': y_range
    }
    
    return coords_normalized, scale_info

# 测试：读取长沙市数据
print("="*80)
print("测试经纬度转换和归一化")
print("="*80)

with open('client_data_by_city/长沙市.json', 'r', encoding='utf-8') as f:
    clients = json.load(f)

print(f"\n数据集: 长沙市")
print(f"客户数: {len(clients)}")

# 提取经纬度
lonlat_coords = np.array([[client['lon'], client['lat']] for client in clients])

print(f"\n原始经纬度范围:")
print(f"  经度: [{lonlat_coords[:, 0].min():.6f}, {lonlat_coords[:, 0].max():.6f}]")
print(f"  纬度: [{lonlat_coords[:, 1].min():.6f}, {lonlat_coords[:, 1].max():.6f}]")

# 计算参考点
lon_ref = np.mean(lonlat_coords[:, 0])
lat_ref = np.mean(lonlat_coords[:, 1])

print(f"\n参考点（区域中心）:")
print(f"  经度: {lon_ref:.6f}")
print(f"  纬度: {lat_ref:.6f}")

# 转换为平面坐标
xy_coords = np.array([lonlat_to_xy(lon, lat, lon_ref, lat_ref) 
                      for lon, lat in lonlat_coords])

print(f"\n平面坐标范围（公里）:")
print(f"  X: [{xy_coords[:, 0].min():.3f}, {xy_coords[:, 0].max():.3f}] km")
print(f"  Y: [{xy_coords[:, 1].min():.3f}, {xy_coords[:, 1].max():.3f}] km")
print(f"  X跨度: {xy_coords[:, 0].max() - xy_coords[:, 0].min():.3f} km")
print(f"  Y跨度: {xy_coords[:, 1].max() - xy_coords[:, 1].min():.3f} km")

# 归一化
coords_norm, scale_info = normalize_coordinates(xy_coords)

print(f"\n归一化后坐标范围:")
print(f"  X: [{coords_norm[:, 0].min():.2f}, {coords_norm[:, 0].max():.2f}]")
print(f"  Y: [{coords_norm[:, 1].min():.2f}, {coords_norm[:, 1].max():.2f}]")

print(f"\n缩放信息:")
print(f"  X范围: {scale_info['x_range']:.3f} km -> 100")
print(f"  Y范围: {scale_info['y_range']:.3f} km -> 100")
print(f"  缩放比例: 1 单位 = {scale_info['x_range']/100:.3f} km (X方向)")
print(f"            1 单位 = {scale_info['y_range']/100:.3f} km (Y方向)")

# 显示几个样本点
print(f"\n样本点对比（前5个）:")
print(f"{'序号':<6} {'经度':<12} {'纬度':<12} {'X(km)':<10} {'Y(km)':<10} {'X_norm':<10} {'Y_norm':<10}")
print("-"*80)
for i in range(min(5, len(clients))):
    print(f"{i:<6} {lonlat_coords[i, 0]:<12.6f} {lonlat_coords[i, 1]:<12.6f} "
          f"{xy_coords[i, 0]:<10.3f} {xy_coords[i, 1]:<10.3f} "
          f"{coords_norm[i, 0]:<10.2f} {coords_norm[i, 1]:<10.2f}")

print("\n" + "="*80)
print("✓ 测试完成")
print("="*80)
