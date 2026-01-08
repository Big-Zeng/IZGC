"""
聚类算法综合对比测试
对比算法:
1. ContourBasedClusteringWithoutZ22 (等值线聚类)
2. DPC-FWSN (MATLAB算法)
3. K-means++
4. DBSCAN
5. CLIQUE (基于网格的聚类)
6. STING (基于网格的统计聚类)

评价指标: AMI, ARI, FMI, Silhouette Score, Inertia
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
import os

import ContourBasedClusteringWithoutZ22 as ContourClustering
from clustering_metrics import ClusteringMetrics

# 尝试导入MATLAB集成
try:
    from matlab_integration import MatlabRunner
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("警告: matlab_integration模块未找到，MATLAB算法将不可用")

# ==================== 配置区域 ====================
# 数据集配置
DATASET = "data/Compound.csv"        # 数据集路径
NORMALIZE_COORDS = True              # 是否归一化坐标到[0,100]范围

# 算法参数
CONTOUR_PARAMS = {
    'grid_size': 3.0,                # 网格大小
    'max_buffer_distance': 0,        # 最大缓冲距离
    'min_cluster_size': 5,           # 最小簇大小
    'density_radius': 30             # 密度计算半径
}

KMEANS_PARAMS = {
    'n_init': 10,
    'random_state': 42,
    'algorithm': 'elkan'             # K-means++
}

DBSCAN_PARAMS = {
    'eps': 5.0,                      # 邻域半径
    'min_samples': 5                 # 最小样本数
}

CLIQUE_PARAMS = {
    'grid_size': 10,                 # 网格大小
    'density_threshold': 3           # 密度阈值
}

STING_PARAMS = {
    'grid_size': 10,                 # 网格大小
    'density_threshold': 3           # 密度阈值
}

MATLAB_PARAMS = {
    'k': 11,                         # DPC-FWSN的近邻数量
}

# 运行控制
RUN_MATLAB = True                    # 是否运行MATLAB算法
SAVE_RESULTS = True                  # 是否保存结果
SHOW_PLOTS = True                    # 是否显示图表
# ==================================================


def normalize_coordinates(X):
    """归一化坐标到[0, 100]范围"""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    X_normalized = np.zeros_like(X)
    X_normalized[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 50
    X_normalized[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 50
    
    return X_normalized, {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
                          'x_range': x_range, 'y_range': y_range}


def calculate_inertia(points, labels):
    """计算簇内误差平方和"""
    inertia = 0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        if len(cluster_points) == 0:
            continue
        cluster_center = np.mean(cluster_points, axis=0)
        inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
    return inertia


def evaluate_clustering(y_true, y_pred, X):
    """评估聚类结果"""
    metrics = ClusteringMetrics.evaluate_all(y_true, y_pred)
    
    # 计算轮廓系数
    if len(set(y_pred)) > 1:
        mask = np.array(y_pred) != -1
        if np.sum(mask) > 0 and len(np.unique(np.array(y_pred)[mask])) > 1:
            silhouette = silhouette_score(X[mask], np.array(y_pred)[mask])
        else:
            silhouette = -1
    else:
        silhouette = -1
    
    # 计算Inertia
    inertia = calculate_inertia(X, np.array(y_pred))
    
    # 统计簇数和噪声点
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    
    return {
        'AMI': metrics['AMI'],
        'ARI': metrics['ARI'],
        'FMI': metrics['FMI'],
        'Silhouette': silhouette,
        'Inertia': inertia,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }


# ==================== 聚类算法实现 ====================

class ClusteringAlgorithms:
    """聚类算法集合"""
    
    def __init__(self, X, y_true):
        self.X = X
        self.y_true = y_true
        self.n_true_clusters = len(np.unique(y_true))
        self.matlab_runner = None
    
    def run_contour_clustering(self, grid_size, max_buffer_distance, min_cluster_size, density_radius):
        """运行等值线聚类算法"""
        print("\n  计算密度...")
        tree = cKDTree(self.X)
        density = np.array([len(tree.query_ball_point(p, density_radius)) for p in self.X])
        
        combined_data = [(*point, density[i], i) for i, point in enumerate(self.X)]
        
        cluster = ContourClustering.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance,
            min_cluster_size=min_cluster_size,
            interactive=False
        )
        
        cluster.perform_clustering()
        labels = cluster.get_predicted_labels()
        
        return np.array(labels)
    
    def run_kmeans(self, n_clusters=None, n_init=10, random_state=42):
        """运行K-means++聚类"""
        if n_clusters is None:
            n_clusters = self.n_true_clusters
        
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, 
                       random_state=random_state, algorithm='elkan')
        labels = kmeans.fit_predict(self.X)
        
        return labels
    
    def run_dbscan(self, eps=0.5, min_samples=5):
        """运行DBSCAN聚类"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(self.X)
        
        return labels
    
    def run_clique(self, grid_size=10, density_threshold=3):
        """
        运行CLIQUE聚类算法（简化实现）
        CLIQUE是一种基于网格的子空间聚类算法
        """
        # 计算网格
        x_min, y_min = self.X.min(axis=0)
        x_max, y_max = self.X.max(axis=0)
        
        n_bins_x = max(1, int((x_max - x_min) / grid_size))
        n_bins_y = max(1, int((y_max - y_min) / grid_size))
        
        # 将点分配到网格
        grid_x = np.clip(((self.X[:, 0] - x_min) / grid_size).astype(int), 0, n_bins_x - 1)
        grid_y = np.clip(((self.X[:, 1] - y_min) / grid_size).astype(int), 0, n_bins_y - 1)
        
        # 计算每个网格的密度
        grid_density = {}
        point_to_grid = {}
        
        for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
            key = (gx, gy)
            if key not in grid_density:
                grid_density[key] = []
            grid_density[key].append(i)
            point_to_grid[i] = key
        
        # 找出密集网格
        dense_grids = {k: v for k, v in grid_density.items() if len(v) >= density_threshold}
        
        # 连接相邻的密集网格形成簇
        labels = np.full(len(self.X), -1)
        cluster_id = 0
        visited = set()
        
        for grid_key in dense_grids:
            if grid_key in visited:
                continue
            
            # BFS找到所有连接的密集网格
            queue = [grid_key]
            cluster_grids = []
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in dense_grids:
                    continue
                
                visited.add(current)
                cluster_grids.append(current)
                
                # 检查8个相邻网格
                cx, cy = current
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (cx + dx, cy + dy)
                        if neighbor in dense_grids and neighbor not in visited:
                            queue.append(neighbor)
            
            # 分配标签
            if cluster_grids:
                for grid_key in cluster_grids:
                    for point_idx in dense_grids[grid_key]:
                        labels[point_idx] = cluster_id
                cluster_id += 1
        
        return labels

    
    def run_sting(self, grid_size=10, density_threshold=3):
        """
        运行STING聚类算法（简化实现）
        STING是一种基于网格的统计信息聚类算法
        """
        # 计算网格
        x_min, y_min = self.X.min(axis=0)
        x_max, y_max = self.X.max(axis=0)
        
        n_bins_x = max(1, int((x_max - x_min) / grid_size))
        n_bins_y = max(1, int((y_max - y_min) / grid_size))
        
        # 将点分配到网格并计算统计信息
        grid_x = np.clip(((self.X[:, 0] - x_min) / grid_size).astype(int), 0, n_bins_x - 1)
        grid_y = np.clip(((self.X[:, 1] - y_min) / grid_size).astype(int), 0, n_bins_y - 1)
        
        # 计算每个网格的统计信息
        grid_stats = {}
        
        for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
            key = (gx, gy)
            if key not in grid_stats:
                grid_stats[key] = {
                    'points': [],
                    'count': 0,
                    'sum_x': 0,
                    'sum_y': 0,
                    'sum_x2': 0,
                    'sum_y2': 0
                }
            
            grid_stats[key]['points'].append(i)
            grid_stats[key]['count'] += 1
            grid_stats[key]['sum_x'] += self.X[i, 0]
            grid_stats[key]['sum_y'] += self.X[i, 1]
            grid_stats[key]['sum_x2'] += self.X[i, 0] ** 2
            grid_stats[key]['sum_y2'] += self.X[i, 1] ** 2
        
        # 计算每个网格的均值和方差
        for key in grid_stats:
            n = grid_stats[key]['count']
            if n > 0:
                mean_x = grid_stats[key]['sum_x'] / n
                mean_y = grid_stats[key]['sum_y'] / n
                var_x = grid_stats[key]['sum_x2'] / n - mean_x ** 2
                var_y = grid_stats[key]['sum_y2'] / n - mean_y ** 2
                grid_stats[key]['mean'] = (mean_x, mean_y)
                grid_stats[key]['var'] = (max(0, var_x), max(0, var_y))
        
        # 找出密集网格（基于统计信息）
        dense_grids = {k: v for k, v in grid_stats.items() if v['count'] >= density_threshold}
        
        # 层次聚合：连接相邻的密集网格
        labels = np.full(len(self.X), -1)
        cluster_id = 0
        visited = set()
        
        for grid_key in dense_grids:
            if grid_key in visited:
                continue
            
            queue = [grid_key]
            cluster_grids = []
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in dense_grids:
                    continue
                
                visited.add(current)
                cluster_grids.append(current)
                
                cx, cy = current
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (cx + dx, cy + dy)
                        if neighbor in dense_grids and neighbor not in visited:
                            queue.append(neighbor)
            
            if cluster_grids:
                for grid_key in cluster_grids:
                    for point_idx in dense_grids[grid_key]['points']:
                        labels[point_idx] = cluster_id
                cluster_id += 1
        
        return labels
    
    def run_matlab_dpc_fwsn(self, k=11):
        """运行MATLAB的DPC-FWSN算法"""
        if not MATLAB_AVAILABLE:
            print("  MATLAB不可用，跳过")
            return None
        
        print("  启动MATLAB...")
        
        # 准备数据 [x, y, label]
        data_with_labels = np.column_stack([self.X, self.y_true])
        
        if self.matlab_runner is None:
            self.matlab_runner = MatlabRunner(method='auto')
        
        labels, metrics = self.matlab_runner.run_algorithm_pr(
            data_with_labels, self.n_true_clusters, k
        )
        
        if labels is None:
            print("  MATLAB算法执行失败")
            return None
        
        return labels
    
    def close_matlab(self):
        """关闭MATLAB引擎"""
        if self.matlab_runner:
            self.matlab_runner.close()


# ==================== 主程序 ====================

def run_comparison(dataset_path, normalize=True):
    """运行算法对比"""
    
    print("=" * 90)
    print("聚类算法综合对比测试")
    print("=" * 90)
    
    # 读取数据
    df = pd.read_csv(dataset_path)
    X = df[['x', 'y']].values.astype(float)
    y_true = df['target'].values
    
    print(f"\n数据集: {dataset_path}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    print(f"X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # 归一化
    X_original = X.copy()
    if normalize:
        X, scale_info = normalize_coordinates(X)
        print(f"\n✓ 坐标已归一化到[0, 100]范围")
    
    # 创建算法实例
    algorithms = ClusteringAlgorithms(X, y_true)
    
    # 存储结果
    results = []
    
    # 1. ContourBasedClustering (等值线聚类)
    print("\n" + "-" * 90)
    print("1. ContourBasedClustering (等值线聚类)")
    print("-" * 90)
    
    start_time = time.time()
    try:
        contour_labels = algorithms.run_contour_clustering(
            grid_size=CONTOUR_PARAMS['grid_size'],
            max_buffer_distance=CONTOUR_PARAMS['max_buffer_distance'],
            min_cluster_size=CONTOUR_PARAMS['min_cluster_size'],
            density_radius=CONTOUR_PARAMS['density_radius']
        )
        contour_time = time.time() - start_time
        contour_metrics = evaluate_clustering(y_true, contour_labels, X)
        
        print(f"  簇数: {contour_metrics['n_clusters']}, 噪声点: {contour_metrics['n_noise']}")
        print(f"  AMI: {contour_metrics['AMI']:.4f}, ARI: {contour_metrics['ARI']:.4f}, FMI: {contour_metrics['FMI']:.4f}")
        print(f"  Silhouette: {contour_metrics['Silhouette']:.4f}, Inertia: {contour_metrics['Inertia']:.2f}")
        print(f"  运行时间: {contour_time:.2f}s")
        
        results.append({
            'name': 'ContourBased',
            'labels': contour_labels,
            'metrics': contour_metrics,
            'time': contour_time
        })
    except Exception as e:
        print(f"  ✗ 运行失败: {e}")
    
    # 2. K-means++
    print("\n" + "-" * 90)
    print("2. K-means++")
    print("-" * 90)
    
    start_time = time.time()
    try:
        kmeans_labels = algorithms.run_kmeans(
            n_clusters=len(np.unique(y_true)),
            n_init=KMEANS_PARAMS['n_init'],
            random_state=KMEANS_PARAMS['random_state']
        )
        kmeans_time = time.time() - start_time
        kmeans_metrics = evaluate_clustering(y_true, kmeans_labels, X)
        
        print(f"  簇数: {kmeans_metrics['n_clusters']}")
        print(f"  AMI: {kmeans_metrics['AMI']:.4f}, ARI: {kmeans_metrics['ARI']:.4f}, FMI: {kmeans_metrics['FMI']:.4f}")
        print(f"  Silhouette: {kmeans_metrics['Silhouette']:.4f}, Inertia: {kmeans_metrics['Inertia']:.2f}")
        print(f"  运行时间: {kmeans_time:.2f}s")
        
        results.append({
            'name': 'K-means++',
            'labels': kmeans_labels,
            'metrics': kmeans_metrics,
            'time': kmeans_time
        })
    except Exception as e:
        print(f"  ✗ 运行失败: {e}")

    
    # 3. DBSCAN
    print("\n" + "-" * 90)
    print("3. DBSCAN")
    print("-" * 90)
    
    start_time = time.time()
    try:
        dbscan_labels = algorithms.run_dbscan(
            eps=DBSCAN_PARAMS['eps'],
            min_samples=DBSCAN_PARAMS['min_samples']
        )
        dbscan_time = time.time() - start_time
        dbscan_metrics = evaluate_clustering(y_true, dbscan_labels, X)
        
        print(f"  簇数: {dbscan_metrics['n_clusters']}, 噪声点: {dbscan_metrics['n_noise']}")
        print(f"  AMI: {dbscan_metrics['AMI']:.4f}, ARI: {dbscan_metrics['ARI']:.4f}, FMI: {dbscan_metrics['FMI']:.4f}")
        print(f"  Silhouette: {dbscan_metrics['Silhouette']:.4f}, Inertia: {dbscan_metrics['Inertia']:.2f}")
        print(f"  运行时间: {dbscan_time:.2f}s")
        
        results.append({
            'name': 'DBSCAN',
            'labels': dbscan_labels,
            'metrics': dbscan_metrics,
            'time': dbscan_time
        })
    except Exception as e:
        print(f"  ✗ 运行失败: {e}")
    
    # 4. CLIQUE
    print("\n" + "-" * 90)
    print("4. CLIQUE (基于网格)")
    print("-" * 90)
    
    start_time = time.time()
    try:
        clique_labels = algorithms.run_clique(
            grid_size=CLIQUE_PARAMS['grid_size'],
            density_threshold=CLIQUE_PARAMS['density_threshold']
        )
        clique_time = time.time() - start_time
        clique_metrics = evaluate_clustering(y_true, clique_labels, X)
        
        print(f"  簇数: {clique_metrics['n_clusters']}, 噪声点: {clique_metrics['n_noise']}")
        print(f"  AMI: {clique_metrics['AMI']:.4f}, ARI: {clique_metrics['ARI']:.4f}, FMI: {clique_metrics['FMI']:.4f}")
        print(f"  Silhouette: {clique_metrics['Silhouette']:.4f}, Inertia: {clique_metrics['Inertia']:.2f}")
        print(f"  运行时间: {clique_time:.2f}s")
        
        results.append({
            'name': 'CLIQUE',
            'labels': clique_labels,
            'metrics': clique_metrics,
            'time': clique_time
        })
    except Exception as e:
        print(f"  ✗ 运行失败: {e}")
    
    # 5. STING
    print("\n" + "-" * 90)
    print("5. STING (基于网格统计)")
    print("-" * 90)
    
    start_time = time.time()
    try:
        sting_labels = algorithms.run_sting(
            grid_size=STING_PARAMS['grid_size'],
            density_threshold=STING_PARAMS['density_threshold']
        )
        sting_time = time.time() - start_time
        sting_metrics = evaluate_clustering(y_true, sting_labels, X)
        
        print(f"  簇数: {sting_metrics['n_clusters']}, 噪声点: {sting_metrics['n_noise']}")
        print(f"  AMI: {sting_metrics['AMI']:.4f}, ARI: {sting_metrics['ARI']:.4f}, FMI: {sting_metrics['FMI']:.4f}")
        print(f"  Silhouette: {sting_metrics['Silhouette']:.4f}, Inertia: {sting_metrics['Inertia']:.2f}")
        print(f"  运行时间: {sting_time:.2f}s")
        
        results.append({
            'name': 'STING',
            'labels': sting_labels,
            'metrics': sting_metrics,
            'time': sting_time
        })
    except Exception as e:
        print(f"  ✗ 运行失败: {e}")
    
    # 6. DPC-FWSN (MATLAB)
    if RUN_MATLAB and MATLAB_AVAILABLE:
        print("\n" + "-" * 90)
        print("6. DPC-FWSN (MATLAB)")
        print("-" * 90)
        
        start_time = time.time()
        try:
            matlab_labels = algorithms.run_matlab_dpc_fwsn(k=MATLAB_PARAMS['k'])
            matlab_time = time.time() - start_time
            
            if matlab_labels is not None:
                matlab_metrics = evaluate_clustering(y_true, matlab_labels, X)
                
                print(f"  簇数: {matlab_metrics['n_clusters']}")
                print(f"  AMI: {matlab_metrics['AMI']:.4f}, ARI: {matlab_metrics['ARI']:.4f}, FMI: {matlab_metrics['FMI']:.4f}")
                print(f"  Silhouette: {matlab_metrics['Silhouette']:.4f}, Inertia: {matlab_metrics['Inertia']:.2f}")
                print(f"  运行时间: {matlab_time:.2f}s")
                
                results.append({
                    'name': 'DPC-FWSN',
                    'labels': matlab_labels,
                    'metrics': matlab_metrics,
                    'time': matlab_time
                })
        except Exception as e:
            print(f"  ✗ 运行失败: {e}")
    else:
        if not RUN_MATLAB:
            print("\n跳过MATLAB算法（RUN_MATLAB=False）")
        elif not MATLAB_AVAILABLE:
            print("\nMATLAB不可用，跳过DPC-FWSN算法")
    
    # 关闭MATLAB
    algorithms.close_matlab()
    
    return results, X_original, y_true


def print_summary(results):
    """打印汇总结果"""
    print("\n" + "=" * 100)
    print("算法对比汇总")
    print("=" * 100)
    
    header = f"{'算法':<15} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'Silhouette':<12} {'Inertia':<12} {'簇数':<8} {'噪声':<8} {'时间(s)':<10}"
    print(header)
    print("-" * 100)
    
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<15} {m['AMI']:<10.4f} {m['ARI']:<10.4f} {m['FMI']:<10.4f} "
              f"{m['Silhouette']:<12.4f} {m['Inertia']:<12.2f} {m['n_clusters']:<8} "
              f"{m['n_noise']:<8} {r['time']:<10.2f}")
    
    print("=" * 100)
    
    # 找出最佳算法
    if results:
        best_ari = max(results, key=lambda x: x['metrics']['ARI'])
        best_ami = max(results, key=lambda x: x['metrics']['AMI'])
        best_fmi = max(results, key=lambda x: x['metrics']['FMI'])
        best_silhouette = max(results, key=lambda x: x['metrics']['Silhouette'])
        
        print("\n最佳算法:")
        print(f"  ARI最高: {best_ari['name']} ({best_ari['metrics']['ARI']:.4f})")
        print(f"  AMI最高: {best_ami['name']} ({best_ami['metrics']['AMI']:.4f})")
        print(f"  FMI最高: {best_fmi['name']} ({best_fmi['metrics']['FMI']:.4f})")
        print(f"  Silhouette最高: {best_silhouette['name']} ({best_silhouette['metrics']['Silhouette']:.4f})")


def plot_comparison(results, X, y_true, save_path=None):
    """绘制对比图"""
    n_plots = len(results) + 1
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 真实标签
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=20, alpha=0.7)
    ax.set_title('Ground Truth')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 各算法结果
    for i, r in enumerate(results):
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        ax = axes[row, col]
        
        ax.scatter(X[:, 0], X[:, 1], c=r['labels'], cmap='tab10', s=20, alpha=0.7)
        ax.set_title(f"{r['name']}\nARI={r['metrics']['ARI']:.3f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # 隐藏多余的子图
    for i in range(len(results) + 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化结果已保存: {save_path}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_results_to_file(results, dataset_name, output_dir='results'):
    """保存结果到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存CSV
    data = []
    for r in results:
        m = r['metrics']
        data.append({
            'Algorithm': r['name'],
            'AMI': m['AMI'],
            'ARI': m['ARI'],
            'FMI': m['FMI'],
            'Silhouette': m['Silhouette'],
            'Inertia': m['Inertia'],
            'n_clusters': m['n_clusters'],
            'n_noise': m['n_noise'],
            'Time': r['time']
        })
    
    df = pd.DataFrame(data)
    csv_path = f"{output_dir}/comparison_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"结果已保存: {csv_path}")
    
    return csv_path


# ==================== 入口点 ====================

if __name__ == "__main__":
    # 运行对比
    results, X_original, y_true = run_comparison(DATASET, normalize=NORMALIZE_COORDS)
    
    # 打印汇总
    print_summary(results)
    
    # 绘制对比图
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    plot_path = f"comparison_{dataset_name}.png" if SAVE_RESULTS else None
    plot_comparison(results, X_original, y_true, save_path=plot_path)
    
    # 保存结果
    if SAVE_RESULTS:
        save_results_to_file(results, dataset_name)
    
    print("\n" + "=" * 90)
    print("对比测试完成！")
    print("=" * 90)
