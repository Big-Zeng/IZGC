"""
聚类算法对比测试 - 简化版
仿照 compareTest.py 的结构
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import ContourBasedClusteringWithoutZ3 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics


# ==================== 配置区域 ====================
# 数据集选择（修改这里来测试不同数据集）
DATASET = "data/flame.csv"          # 数据集路径
N_CLUSTERS = 2                       # 真实簇数

# 算法参数
KMEANS_PARAMS = {
    'n_clusters': N_CLUSTERS,
    'random_state': 42,
    'n_init': 10
}

DBSCAN_PARAMS = {
    'eps': 1.5,                      # 邻域半径
    'min_samples': 5                 # 最小样本数
}

CONTOUR_PARAMS = {
    'grid_size': None,               # None表示自动计算，或指定具体值如1.2
    'max_buffer_distance': 5,        # 最大缓冲距离
    'auto_grid_multiplier': 2.0,     # 自动计算时：格子数 = 点数 × 此倍数
    'optimize_grid': False           # True表示使用梯度下降优化grid_size
}
# ==================================================


class VRPAnalyzer:
    """聚类分析器"""
    
    def __init__(self, coords, true_labels):
        self.coords = coords
        self.true_labels = true_labels
    
    def calculate_auto_grid_size(self, grid_multiplier=2.0):
        """
        自动计算grid_size
        使得格子数量约为点数的grid_multiplier倍
        
        参数:
            grid_multiplier: 格子数量倍数（默认2.0，即格子数=点数×2）
            
        返回:
            grid_size: 计算得到的网格大小
        """
        n_points = len(self.coords)
        
        # 计算包裹所有点的矩形
        x_min, y_min = self.coords.min(axis=0)
        x_max, y_max = self.coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # 矩形面积
        area = width * height
        
        # 目标格子数量
        target_grid_count = n_points * grid_multiplier
        
        # 计算grid_size
        grid_size = np.sqrt(area / target_grid_count)
        
        print(f"\n自动计算grid_size:")
        print(f"  点数: {n_points}")
        print(f"  矩形范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
        print(f"  矩形尺寸: {width:.2f} × {height:.2f}")
        print(f"  矩形面积: {area:.2f}")
        print(f"  目标格子数: {target_grid_count:.0f} (点数 × {grid_multiplier})")
        print(f"  计算得到的grid_size: {grid_size:.3f}")
        print(f"  实际格子数约: {int(width/grid_size) * int(height/grid_size)}")
        
        return grid_size
    
    def calculate_inertia(self, points, labels):
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
    
    def run_kmeans_clustering(self, n_clusters):
        """KMeans聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.coords)
        
        inertia = self.calculate_inertia(self.coords, labels)
        silhouette_avg = silhouette_score(self.coords, labels) if len(set(labels)) > 1 else -1
        
        # 使用自定义评估指标
        metrics = ClusteringMetrics.evaluate_all(self.true_labels, labels)
        
        return kmeans, labels, inertia, silhouette_avg, metrics['AMI'], metrics['ARI'], metrics['FMI']
    
    def run_dbscan_clustering(self, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(self.coords)
        
        inertia = self.calculate_inertia(self.coords, labels)
        
        # 计算轮廓系数（排除噪声点）
        if len(set(labels)) > 1:
            mask = labels != -1
            if np.sum(mask) > 0 and len(np.unique(labels[mask])) > 1:
                silhouette_avg = silhouette_score(self.coords[mask], labels[mask])
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
        
        # 使用自定义评估指标
        metrics = ClusteringMetrics.evaluate_all(self.true_labels, labels)
        
        return dbscan, labels, inertia, silhouette_avg, metrics['AMI'], metrics['ARI'], metrics['FMI']
    
    def run_contour_clustering(self, grid_size=None, max_buffer_distance=5, auto_grid_multiplier=2.0, optimize_grid=False):
        """
        等值线聚类
        
        参数:
            grid_size: 网格大小，如果为None则自动计算
            max_buffer_distance: 最大缓冲距离
            auto_grid_multiplier: 自动计算时的格子数量倍数（默认2.0）
            optimize_grid: 是否使用梯度下降优化grid_size
        """
        # 如果启用优化
        if optimize_grid:
            print("\n使用梯度下降优化grid_size...")
            from optimize_grid_size import GridSizeOptimizer
            
            optimizer = GridSizeOptimizer(self.coords, self.true_labels, max_buffer_distance)
            grid_size, _, _ = optimizer.optimize(
                initial_step=None,
                min_step=0.01,
                step_decay=0.5,
                max_iterations=15,
                tolerance=0.001,
                min_grid_size=0.1,
                verbose=True
            )
            print(f"优化后的grid_size: {grid_size:.3f}\n")
        
        # 如果未指定grid_size且未优化，则自动计算
        elif grid_size is None:
            grid_size = self.calculate_auto_grid_size(grid_multiplier=auto_grid_multiplier)
        else:
            print(f"\n使用指定的grid_size: {grid_size:.3f}")
        
        # 计算密度（使用Voronoi）
        points = self.coords
        vor = Voronoi(points)
        density = np.zeros(len(points))
        
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:
                density[i] = 0
                continue
            
            polygon = vor.vertices[region]
            x = polygon[:, 0]
            y = polygon[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            density[i] = 1.0 / area if area > 0 else 0
        
        # 准备数据
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
        
        # 运行聚类
        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance
        )
        
        final_points = cluster.perform_clustering()
        cluster.visualize_clusters()
        
        pre_labels = cluster.get_predicted_labels()
        
        # 计算评估指标
        inertia = self.calculate_inertia(self.coords, pre_labels)
        
        if len(set(pre_labels)) > 1:
            mask = np.array(pre_labels) != -1
            if np.sum(mask) > 0 and len(np.unique(np.array(pre_labels)[mask])) > 1:
                silhouette_avg = silhouette_score(self.coords[mask], np.array(pre_labels)[mask])
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
        
        # 使用自定义评估指标
        metrics = ClusteringMetrics.evaluate_all(self.true_labels, pre_labels)
        
        return final_points, pre_labels, inertia, silhouette_avg, metrics['AMI'], metrics['ARI'], metrics['FMI']


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("聚类算法对比测试")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv(DATASET)
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    print(f"\n数据集: {DATASET}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    print(f"X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # 创建分析器
    analyzer = VRPAnalyzer(X, y_true)
    
    # 1. KMeans聚类
    print("\n" + "-" * 80)
    print("1. KMeans 聚类")
    print("-" * 80)
    kmeans_model, kmeans_labels, kmeans_inertia, kmeans_silhouette, kmeans_ami, kmeans_ari, kmeans_fmi = \
        analyzer.run_kmeans_clustering(KMEANS_PARAMS['n_clusters'])
    
    print(f"Inertia:    {kmeans_inertia:.4f}")
    print(f"Silhouette: {kmeans_silhouette:.4f}")
    print(f"AMI:        {kmeans_ami:.4f}")
    print(f"ARI:        {kmeans_ari:.4f}")
    print(f"FMI:        {kmeans_fmi:.4f}")
    
    # 2. DBSCAN聚类
    print("\n" + "-" * 80)
    print("2. DBSCAN 聚类")
    print("-" * 80)
    dbscan_model, dbscan_labels, dbscan_inertia, dbscan_silhouette, dbscan_ami, dbscan_ari, dbscan_fmi = \
        analyzer.run_dbscan_clustering(eps=DBSCAN_PARAMS['eps'], min_samples=DBSCAN_PARAMS['min_samples'])
    
    n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"发现簇数: {n_clusters_found}, 噪声点: {n_noise}")
    print(f"Inertia:    {dbscan_inertia:.4f}")
    print(f"Silhouette: {dbscan_silhouette:.4f}")
    print(f"AMI:        {dbscan_ami:.4f}")
    print(f"ARI:        {dbscan_ari:.4f}")
    print(f"FMI:        {dbscan_fmi:.4f}")
    
    # 3. Contour-Based聚类
    print("\n" + "-" * 80)
    print("3. Contour-Based 聚类")
    print("-" * 80)
    contour_model, contour_labels, contour_inertia, contour_silhouette, contour_ami, contour_ari, contour_fmi = \
        analyzer.run_contour_clustering(
            grid_size=CONTOUR_PARAMS['grid_size'],
            max_buffer_distance=CONTOUR_PARAMS['max_buffer_distance'],
            auto_grid_multiplier=CONTOUR_PARAMS['auto_grid_multiplier'],
            optimize_grid=CONTOUR_PARAMS['optimize_grid']
        )
    
    n_clusters_found = len(set(contour_labels)) - (1 if -1 in contour_labels else 0)
    n_noise = list(contour_labels).count(-1)
    print(f"发现簇数: {n_clusters_found}, 噪声点: {n_noise}")
    print(f"Inertia:    {contour_inertia:.4f}")
    print(f"Silhouette: {contour_silhouette:.4f}")
    print(f"AMI:        {contour_ami:.4f}")
    print(f"ARI:        {contour_ari:.4f}")
    print(f"FMI:        {contour_fmi:.4f}")
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("算法对比汇总")
    print("=" * 80)
    print(f"{'算法':<20} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'Silhouette':<12} {'Inertia':<12}")
    print("-" * 80)
    print(f"{'KMeans':<20} {kmeans_ami:<10.4f} {kmeans_ari:<10.4f} {kmeans_fmi:<10.4f} "
          f"{kmeans_silhouette:<12.4f} {kmeans_inertia:<12.2f}")
    print(f"{'DBSCAN':<20} {dbscan_ami:<10.4f} {dbscan_ari:<10.4f} {dbscan_fmi:<10.4f} "
          f"{dbscan_silhouette:<12.4f} {dbscan_inertia:<12.2f}")
    print(f"{'Contour-Based':<20} {contour_ami:<10.4f} {contour_ari:<10.4f} {contour_fmi:<10.4f} "
          f"{contour_silhouette:<12.4f} {contour_inertia:<12.2f}")
    print("=" * 80)
    
    # 可视化对比
    fig = plt.figure(figsize=(16, 4))
    
    # 真实标签
    plt.subplot(141)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=20)
    plt.title('Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # KMeans
    plt.subplot(142)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=20)
    plt.title(f'KMeans\nARI={kmeans_ari:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # DBSCAN
    plt.subplot(143)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='tab10', s=20)
    plt.title(f'DBSCAN\nARI={dbscan_ari:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Contour-Based
    plt.subplot(144)
    plt.scatter(X[:, 0], X[:, 1], c=contour_labels, cmap='tab10', s=20)
    plt.title(f'Contour-Based\nARI={contour_ari:.3f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()
