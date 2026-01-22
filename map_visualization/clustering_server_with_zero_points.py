"""
Flask服务器 - 支持手动添加零值点的交互式聚类分析
功能：
1. 加载真实数据（长沙市等）
2. 在地图上手动点击添加零值点
3. 保存零值点到CSV文件
4. 结合零值点执行聚类
5. 在地图上显示聚类结果和零值点
6. 坐标归一化到[0, 100]范围（提高聚类效果）
"""

from flask import Flask, request, jsonify, send_from_directory, make_response
import json
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
from functools import wraps

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ContourBasedClusteringWithoutZ22 as ContourClustering
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from algorithm_comparison.utils import normalize_coordinates

app = Flask(__name__)

# 手动实现CORS支持
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)

# 全局变量
data_points = None  # 原始数据点
zero_points = []    # 手动添加的零值点
current_dataset = None  # 当前数据集名称

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'real_data')


def load_data(dataset_name):
    """加载数据集"""
    global data_points, current_dataset, zero_points
    
    filepath = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(filepath):
        # 尝试带_real后缀
        filepath = os.path.join(DATA_DIR, f"{dataset_name}_real.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath)
    data_points = df[['x', 'y']].values.tolist()
    current_dataset = dataset_name
    
    # 尝试加载已保存的零值点
    zero_points_file = os.path.join(DATA_DIR, f"{dataset_name}_zero_points.csv")
    if os.path.exists(zero_points_file):
        zero_df = pd.read_csv(zero_points_file)
        zero_points = zero_df[['x', 'y']].values.tolist()
        print(f"已加载 {len(zero_points)} 个已保存的零值点")
    else:
        zero_points = []
    
    return data_points


@app.route('/')
def index():
    """返回主页面"""
    return send_from_directory('.', 'interactive_clustering_leaflet_with_zero_points.html')


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """获取可用的数据集列表"""
    try:
        datasets = []
        for f in os.listdir(DATA_DIR):
            if f.endswith('_real.csv'):
                name = f.replace('_real.csv', '')
                datasets.append(name)
            elif f.endswith('.csv') and not f.endswith('_zero_points.csv'):
                name = f.replace('.csv', '')
                if name not in datasets:
                    datasets.append(name)
        
        return jsonify({
            'success': True,
            'datasets': sorted(datasets)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/load_data', methods=['POST'])
def api_load_data():
    """加载指定数据集"""
    try:
        params = request.json
        dataset_name = params.get('dataset', '长沙市')
        
        points = load_data(dataset_name)
        
        print(f"✓ 加载数据集: {dataset_name}, 共 {len(points)} 个点")
        
        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'points': points,
            'zero_points': zero_points,
            'count': len(points),
            'zero_count': len(zero_points)
        })
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/add_zero_point', methods=['POST'])
def add_zero_point():
    """添加一个零值点"""
    global zero_points
    
    try:
        params = request.json
        x = float(params.get('x'))
        y = float(params.get('y'))
        
        zero_points.append([x, y])
        print(f"✓ 添加零值点: ({x:.6f}, {y:.6f}), 当前共 {len(zero_points)} 个")
        
        return jsonify({
            'success': True,
            'zero_points': zero_points,
            'count': len(zero_points)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/add_zero_points_batch', methods=['POST'])
def add_zero_points_batch():
    """批量添加零值点并自动保存到CSV"""
    global zero_points, current_dataset
    
    try:
        params = request.json
        points = params.get('points', [])
        auto_save = params.get('auto_save', True)
        
        if not points:
            return jsonify({
                'success': False,
                'error': '没有提供零值点数据'
            }), 400
        
        # 批量添加
        added_count = 0
        for p in points:
            x = float(p[0])
            y = float(p[1])
            zero_points.append([x, y])
            added_count += 1
        
        print(f"✓ 批量添加 {added_count} 个零值点, 当前共 {len(zero_points)} 个")
        
        # 自动保存到CSV
        saved_file = None
        if auto_save and current_dataset:
            filename = f"{current_dataset}_zero_points.csv"
            filepath = os.path.join(DATA_DIR, filename)
            df = pd.DataFrame(zero_points, columns=['x', 'y'])
            df.to_csv(filepath, index=False)
            saved_file = filename
            print(f"✓ 零值点已自动保存到: {filepath}")
        
        return jsonify({
            'success': True,
            'zero_points': zero_points,
            'count': len(zero_points),
            'added_count': added_count,
            'saved_file': saved_file
        })
    except Exception as e:
        print(f"✗ 批量添加零值点失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/remove_zero_point', methods=['POST'])
def remove_zero_point():
    """移除一个零值点"""
    global zero_points
    
    try:
        params = request.json
        index = int(params.get('index', -1))
        
        if 0 <= index < len(zero_points):
            removed = zero_points.pop(index)
            print(f"✓ 移除零值点: ({removed[0]:.6f}, {removed[1]:.6f})")
        
        return jsonify({
            'success': True,
            'zero_points': zero_points,
            'count': len(zero_points)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear_zero_points', methods=['POST'])
def clear_zero_points():
    """清空所有零值点"""
    global zero_points
    
    zero_points = []
    print("✓ 已清空所有零值点")
    
    return jsonify({
        'success': True,
        'zero_points': [],
        'count': 0
    })


@app.route('/api/save_zero_points', methods=['POST'])
def save_zero_points():
    """保存零值点到CSV文件"""
    global zero_points, current_dataset
    
    try:
        if not current_dataset:
            return jsonify({
                'success': False,
                'error': '请先加载数据集'
            }), 400
        
        if not zero_points:
            return jsonify({
                'success': False,
                'error': '没有零值点需要保存'
            }), 400
        
        # 保存到CSV
        filename = f"{current_dataset}_zero_points.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        df = pd.DataFrame(zero_points, columns=['x', 'y'])
        df.to_csv(filepath, index=False)
        
        print(f"✓ 零值点已保存到: {filepath}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'count': len(zero_points)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/proxy_tile', methods=['GET'])
def proxy_tile():
    """代理百度地图瓦片，解决跨域问题"""
    import requests
    
    try:
        tile_url = request.args.get('url')
        if not tile_url:
            return jsonify({'success': False, 'error': '缺少url参数'}), 400
        
        # 只允许百度地图域名
        if not any(domain in tile_url for domain in ['baidu.com', 'bdimg.com', 'bdstatic.com']):
            return jsonify({'success': False, 'error': '不允许的域名'}), 403
        
        # 请求瓦片
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://map.baidu.com/'
        }
        resp = requests.get(tile_url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            response = make_response(resp.content)
            response.headers['Content-Type'] = resp.headers.get('Content-Type', 'image/png')
            response.headers['Cache-Control'] = 'max-age=86400'
            return response
        else:
            return jsonify({'success': False, 'error': f'请求失败: {resp.status_code}'}), resp.status_code
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export_map', methods=['POST'])
def export_map():
    """服务器端导出地图图片"""
    import requests
    from io import BytesIO
    from PIL import Image
    import math
    
    try:
        params = request.json
        bounds = params.get('bounds')  # {minLng, maxLng, minLat, maxLat}
        width = params.get('width', 1024)
        height = params.get('height', 768)
        scale = params.get('scale', 2)
        
        if not bounds:
            return jsonify({'success': False, 'error': '缺少bounds参数'}), 400
        
        # 计算中心点
        center_lng = (bounds['minLng'] + bounds['maxLng']) / 2
        center_lat = (bounds['minLat'] + bounds['maxLat']) / 2
        
        # 计算缩放级别
        lng_range = bounds['maxLng'] - bounds['minLng']
        lat_range = bounds['maxLat'] - bounds['minLat']
        max_range = max(lng_range, lat_range)
        
        if max_range > 1: zoom = 8
        elif max_range > 0.5: zoom = 9
        elif max_range > 0.2: zoom = 10
        elif max_range > 0.1: zoom = 11
        elif max_range > 0.05: zoom = 12
        elif max_range > 0.02: zoom = 13
        else: zoom = 14
        
        # 使用百度静态图API
        ak = '7GJV8YHkODN1xUoziJd3jvkrIYmT05Oc'
        static_url = f"https://api.map.baidu.com/staticimage/v2?ak={ak}&center={center_lng},{center_lat}&width={width}&height={height}&zoom={zoom}&scale={scale}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://map.baidu.com/'
        }
        
        resp = requests.get(static_url, headers=headers, timeout=30)
        
        if resp.status_code == 200:
            # 返回图片的base64
            import base64
            img_base64 = base64.b64encode(resp.content).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'width': width * scale,
                'height': height * scale,
                'zoom': zoom,
                'center': [center_lng, center_lat]
            })
        else:
            return jsonify({'success': False, 'error': f'获取地图失败: {resp.status_code}'}), 500
            
    except Exception as e:
        print(f"导出地图失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def denormalize_coordinates(X_norm, x_min, x_max, y_min, y_max):
    """
    将归一化坐标转换回原始坐标
    
    参数:
        X_norm: 归一化后的坐标数组 (n, 2)，范围[0, 100]
        x_min, x_max: 原始x坐标范围
        y_min, y_max: 原始y坐标范围
    
    返回:
        原始坐标数组
    """
    X_norm = np.array(X_norm)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    X_orig = np.zeros_like(X_norm)
    X_orig[:, 0] = X_norm[:, 0] / 100 * x_range + x_min if x_range > 0 else x_min
    X_orig[:, 1] = X_norm[:, 1] / 100 * y_range + y_min if y_range > 0 else y_min
    
    return X_orig


@app.route('/api/cluster', methods=['POST'])
def perform_clustering():
    """执行聚类分析（结合零值点）"""
    global data_points, zero_points
    
    try:
        if not data_points:
            return jsonify({
                'success': False,
                'error': '请先加载数据'
            }), 400
        
        # 获取参数
        params = request.json
        grid_size = float(params.get('grid_size', 0.01))
        max_buffer_distance = int(params.get('max_buffer_distance', 5))
        min_cluster_size = int(params.get('min_cluster_size', 5))
        merge_small_clusters = params.get('merge_small_clusters', True)
        use_normalization = params.get('normalize', True)  # 默认启用归一化
        use_zero_points = params.get('use_zero_points', True)  # 是否使用零值点
        
        # 根据参数决定实际使用的零值点
        actual_zero_points = zero_points if use_zero_points else []
        
        print(f"\n{'='*80}")
        print(f"收到聚类请求:")
        print(f"  grid_size: {grid_size:.3f}")
        print(f"  max_buffer_distance: {max_buffer_distance}")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"  merge_small_clusters: {merge_small_clusters}")
        print(f"  normalize: {use_normalization}")
        print(f"  use_zero_points: {use_zero_points}")
        print(f"  数据点数: {len(data_points)}")
        print(f"  零值点数: {len(actual_zero_points)} (总共{len(zero_points)}个)")
        print(f"{'='*80}\n")
        
        # 准备原始坐标
        coords_orig = np.array(data_points)
        
        # 保存原始坐标范围（用于反归一化）
        x_min, y_min = coords_orig.min(axis=0)
        x_max, y_max = coords_orig.max(axis=0)
        
        # 坐标归一化到[0, 100]范围
        if use_normalization:
            coords = normalize_coordinates(coords_orig)
            print(f"✓ 坐标已归一化到[0, 100]")
            print(f"  原始范围: x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")
        else:
            coords = coords_orig
        
        # 计算密度作为z值（使用归一化后的坐标）
        tree = cKDTree(coords)
        # 密度计算半径也需要根据归一化调整
        density_radius = 1.0 if use_normalization else 0.01
        density = np.array([len(tree.query_ball_point(p, density_radius)) for p in coords])
        
        # 归一化密度到合理范围
        if density.max() > 0:
            density = density / density.max() * 10
        
        combined_data = []
        for i, (x, y) in enumerate(coords):
            combined_data.append((x, y, float(density[i]), i))
        
        # 准备零值点（手动添加的虚拟点）- 也需要归一化
        manual_virtual_points = []
        if actual_zero_points:
            zero_coords_orig = np.array(actual_zero_points)
            if use_normalization:
                # 使用相同的归一化参数
                x_range = x_max - x_min
                y_range = y_max - y_min
                zero_coords = np.zeros_like(zero_coords_orig)
                zero_coords[:, 0] = ((zero_coords_orig[:, 0] - x_min) / x_range * 100) if x_range > 0 else 50
                zero_coords[:, 1] = ((zero_coords_orig[:, 1] - y_min) / y_range * 100) if y_range > 0 else 50
            else:
                zero_coords = zero_coords_orig
            
            for x, y in zero_coords:
                manual_virtual_points.append((x, y, 0.0, -2))  # -2表示手动添加
        
        print(f"准备聚类数据: {len(combined_data)} 个数据点, {len(manual_virtual_points)} 个零值点")
        
        # 创建聚类实例（禁用交互式界面）
        import matplotlib
        matplotlib.use('Agg')
        
        cluster = ContourClustering.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance,
            manual_virtual_points=manual_virtual_points,
            min_cluster_size=min_cluster_size,
            interactive=False,
            merge_small_clusters=merge_small_clusters
        )
        
        # 执行聚类
        cluster.perform_clustering()
        print("✓ 聚类完成")
        
        # 获取聚类标签
        pred_labels = cluster.get_predicted_labels()
        pred_labels_array = np.array(pred_labels)
        
        # 统计聚类结果
        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_noise = list(pred_labels).count(-1)
        
        print(f"发现簇数: {n_clusters}")
        print(f"噪声点数: {n_noise}")
        
        # 计算评价指标（使用归一化后的坐标）
        metrics = {}
        mask = pred_labels_array != -1
        coords_valid = coords[mask]
        labels_valid = pred_labels_array[mask]
        
        if n_clusters > 1 and len(labels_valid) > 0:
            try:
                silhouette = silhouette_score(coords_valid, labels_valid, metric='euclidean')
                ch_score = calinski_harabasz_score(coords_valid, labels_valid)
                db_score = davies_bouldin_score(coords_valid, labels_valid)
                
                metrics = {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(ch_score),
                    'davies_bouldin_score': float(db_score)
                }
            except Exception as e:
                print(f"计算指标失败: {e}")
        
        # 获取等值线多边形数据（需要转换回原始坐标用于地图显示）
        contour_polygons = []
        for i, center in enumerate(cluster.density_centers):
            cluster_points = [point for point, centers in cluster.point_to_density_center.items() 
                             if center in centers]
            if len(cluster_points) == 0:
                continue
            
            # 获取归一化坐标下的多边形
            polygon_norm = center.polygon
            
            # 转换回原始坐标
            if use_normalization:
                polygon_orig = denormalize_coordinates(polygon_norm, x_min, x_max, y_min, y_max)
                polygon_coords = polygon_orig.tolist()
                center_norm = np.array([center.center])
                center_orig = denormalize_coordinates(center_norm, x_min, x_max, y_min, y_max)
                center_coords = center_orig[0].tolist()
            else:
                polygon_coords = polygon_norm.tolist()
                center_coords = list(center.center)
            
            contour_polygons.append({
                'cluster_id': i,
                'polygon': polygon_coords,
                'center': center_coords,
                'point_count': len(cluster_points)
            })
        
        result = {
            'success': True,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_points': len(data_points),
            'n_zero_points': len(actual_zero_points),
            'metrics': metrics,
            'labels': pred_labels,
            'contour_polygons': contour_polygons,
            'normalized': use_normalization,
            'used_zero_points': use_zero_points
        }
        
        print(f"✓ 聚类分析完成")
        if metrics:
            print(f"  轮廓系数: {metrics.get('silhouette_score', 'N/A'):.4f}")
        print(f"{'='*80}\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"✗ 聚类失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("交互式聚类分析服务器（支持零值点）")
    print("="*80)
    print("功能:")
    print("  1. 加载真实数据（长沙市等城市数据）")
    print("  2. 在地图上手动点击添加零值点")
    print("  3. 保存零值点到CSV文件")
    print("  4. 结合零值点执行聚类")
    print("  5. 在地图上显示聚类结果和零值点")
    print("  6. 坐标归一化到[0, 100]范围（提高聚类效果）")
    print("="*80)
    print("服务器启动中...")
    print("访问地址: http://localhost:5001")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
