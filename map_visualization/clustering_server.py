"""
Flask服务器 - 提供交互式聚类分析API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ContourBasedClusteringWithoutZ22_map as ContourClustering
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variable holding the customer data
clients_data = None

def load_client_data():
    """加载客户数据"""
    global clients_data
    if clients_data is None:
        with open('client_data.json', 'r', encoding='utf-8') as f:
            clients_data = json.load(f)
    return clients_data

@app.route('/')
def index():
    """返回主页面"""
    return send_from_directory('.', 'interactive_clustering_baidu.html')

@app.route('/api/cluster', methods=['POST'])
def perform_clustering():
    """执行聚类分析API"""
    try:
        # Get the parameters
        params = request.json
        grid_size = float(params.get('grid_size', 0.05))
        max_buffer_distance = int(params.get('max_buffer_distance', 5))
        min_cluster_size = int(params.get('min_cluster_size', 10))
        
        print(f"\n{'='*80}")
        print(f"收到聚类请求:")
        print(f"  grid_size: {grid_size}")
        print(f"  max_buffer_distance: {max_buffer_distance}")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"{'='*80}\n")
        
        # Load the customer data
        clients = load_client_data()
        print(f"✓ 加载了 {len(clients)} 个客户")
        
        # Extract the coordinate data
        coords = np.array([[client['lon'], client['lat']] for client in clients])
        
        # Compute density
        print("正在计算点密度...")
        tree = cKDTree(coords)
        density = np.array([len(tree.query_ball_point(p, 0.1)) for p in coords])
        print(f"✓ 密度计算完成")
        
        # Prepare the clustering data
        combined_data = []
        for i, client in enumerate(clients):
            combined_data.append((
                client['lon'],
                client['lat'],
                density[i],
                i
            ))
        
        # Create the clustering instance with the interactive UI disabled
        print("正在执行聚类...")
        cluster = ContourClustering.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance,
            min_cluster_size=min_cluster_size
        )
        
        # Run clustering; the algorithm must support non-interactive mode
        # The original approach is kept, but matplotlib's interactive display is switched off
        import matplotlib
        matplotlib.use('Agg')  # Use a non-interactive backend
        
        cluster.perform_clustering()
        print("✓ 聚类完成")
        
        # Get the cluster labels
        pred_labels = cluster.get_predicted_labels()
        pred_labels_array = np.array(pred_labels)
        
        # Summarise the clustering results
        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_noise = list(pred_labels).count(-1)
        
        print(f"发现簇数: {n_clusters}")
        print(f"噪声点数: {n_noise}")
        
        # Count the points of each cluster
        cluster_counts = {}
        for label in pred_labels:
            if label != -1:
                cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        # Compute the metrics
        metrics = {}
        mask = pred_labels_array != -1
        coords_valid = coords[mask]
        labels_valid = pred_labels_array[mask]
        
        if n_clusters > 1 and len(labels_valid) > 0:
            # Silhouette coefficient
            silhouette = silhouette_score(coords_valid, labels_valid, metric='euclidean')
            
            # Calinski-Harabasz index
            ch_score = calinski_harabasz_score(coords_valid, labels_valid)
            
            # Davies-Bouldin index
            db_score = davies_bouldin_score(coords_valid, labels_valid)
            
            # Within-cluster sum of squares
            inertia = 0.0
            for label in np.unique(labels_valid):
                cluster_points = coords_valid[labels_valid == label]
                cluster_center = np.mean(cluster_points, axis=0)
                inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
            
            # Inter-cluster distance
            cluster_centers = []
            for label in np.unique(labels_valid):
                cluster_points = coords_valid[labels_valid == label]
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
            
            cluster_centers = np.array(cluster_centers)
            if len(cluster_centers) > 1:
                from scipy.spatial.distance import pdist
                inter_cluster_distances = pdist(cluster_centers)
                
                metrics = {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(ch_score),
                    'davies_bouldin_score': float(db_score),
                    'inertia': float(inertia),
                    'min_inter_cluster_distance': float(inter_cluster_distances.min()),
                    'max_inter_cluster_distance': float(inter_cluster_distances.max()),
                    'avg_inter_cluster_distance': float(inter_cluster_distances.mean())
                }
        
        # Prepare the return payload
        # Get the contour polygon data
        contour_polygons = []
        for i, center in enumerate(cluster.density_centers):
            # Get the point count of this cluster
            cluster_points = [point for point, centers in cluster.point_to_density_center.items() 
                             if center in centers]
            if len(cluster_points) == 0:
                continue
            
            # Convert the polygon coordinates to list form
            polygon_coords = center.polygon.tolist()
            
            contour_polygons.append({
                'cluster_id': i,
                'polygon': polygon_coords,  # [[lon, lat], [lon, lat], ...]
                'center': list(center.center),  # [lon, lat]
                'point_count': len(cluster_points)
            })
        
        result = {
            'success': True,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_valid_points': len(labels_valid),
            'cluster_counts': cluster_counts,
            'metrics': metrics,
            'labels': pred_labels,  # Cluster label of each customer
            'contour_polygons': contour_polygons  # Contour polygon data
        }
        
        print(f"✓ 聚类分析完成")
        print(f"  簇数: {n_clusters}")
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

@app.route('/api/clients', methods=['GET'])
def get_clients():
    """获取客户数据"""
    try:
        clients = load_client_data()
        return jsonify({
            'success': True,
            'clients': clients
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("交互式聚类分析服务器")
    print("="*80)
    print("服务器启动中...")
    print("访问地址: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
