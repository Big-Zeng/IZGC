"""
使用保存的MATLAB结果进行对比测试
不需要Python 3.10或MATLAB Engine
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
from load_matlab_results import get_matlab_result, list_available_datasets, print_matlab_results_summary


# ==================== 配置区域 ====================
DATASET = "flame"                    # 数据集名称
N_CLUSTERS = 2                       # 真实簇数

# 算法参数
KMEANS_PARAMS = {
    'n_clusters': N_CLUSTERS,
    'random_state': 42,
    'n_init': 10
}

DBSCAN_PARAMS = {
    'eps': 1.5,
    'min_samples': 5
}

CONTOUR_PARAMS = {
    'grid_size': 1.2,
    'max_buffer_distance': 5
}

# 是否包含MATLAB结果
INCLUDE_MATLAB = True                # 设置为False可跳过MATLAB
# ==================================================


class VRPAnalyzer:
    """聚类分析器"""
    
    def __init__(self, coords, true_labels):
        self.coords = coords
        self.true_labels = true_labels
    
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
        
        metrics = ClusteringMetrics.evaluate_all(self.true_labels, labels)
        
        return kmeans, labels, inertia, silhouette_avg, metrics['AMI'], metrics['ARI'], metrics['FMI']
    
    def run_dbscan_clustering(self, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(self.coords)
        
        inertia = self.calculate_inertia(self.coords, labels)
        
        if len(set(labels)) > 1:
            mask = labels != -1
            if np.sum(mask) > 0 and len(np.unique(labels[mask])) > 1:
                silhouette_avg = silhouette_score(self.coords[mask], labels[mask])
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
        
        metrics = ClusteringMetrics.evaluate_all(self.true_labels, labels)
        
        return dbscan, labels, inertia, silhouette_avg, metrics['AMI'], metrics['ARI'], metrics['FMI']
    
    def run_contour_clustering(self, grid_size=1.2, max_buffer_distance=5):
        """等值线聚类"""
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
        
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
        
        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance
        )
        
        final_points = cluster.perform_clustering()
        cluster.visualize_clusters()
        
        pre_labels = cluster.get_predicted_labels()
        
        inertia = self.calculate_inertia(self.coords, pre_labels)
        
        if len(set(pre_labels)) > 1:
            mask = np.array(pre_labels) != -1
            if np.sum(mask) > 0 and len(np.unique(np.array(pre_labels)[mask])) > 1:
                silhouette_avg = silhouette_score(self.coords[mask], np.array(pre_labels)[mask])
            else:
                silhouette_avg = -1
        else:
            silhouette_avg = -1
        
        metrics = ClusteringMetrics.evaluate_all(self.true_labels, pre_labels)
        
        return final_points, pre_labels, inertia, silhouette_avg, metrics['AMI'], metrics['ARI'], metrics['FMI']


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("聚类算法对比测试（使用保存的MATLAB结果）")
    print("=" * 80)
    
    # 检查MATLAB结果是否可用
    available_datasets = list_available_datasets()
    if not available_datasets:
        print("\n警告: 未找到MATLAB结果文件")
        print("请先在MATLAB中运行: compare/algorithm/run_and_save_results.m")
        print("\n将继续运行，但不包含MATLAB结果\n")
        INCLUDE_MATLAB = False
    else:
        print(f"\n找到 {len(available_datasets)} 个MATLAB结果:")
        for name in available_datasets:
            print(f"  - {name}")
        print()
    
    # 读取数据
    df = pd.read_csv(f"data/{DATASET}.csv")
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    print(f"数据集: {DATASET}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    print(f"X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # 创建分析器
    analyzer = VRPAnalyzer(X, y_true)
    
    results = []
    
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
    
    results.append({
        'name': 'KMeans',
        'labels': kmeans_labels,
        'ami': kmeans_ami,
        'ari': kmeans_ari,
        'fmi': kmeans_fmi,
        'silhouette': kmeans_silhouette,
        'inertia': kmeans_inertia
    })
    
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
    
    results.append({
        'name': 'DBSCAN',
        'labels': dbscan_labels,
        'ami': dbscan_ami,
        'ari': dbscan_ari,
        'fmi': dbscan_fmi,
        'silhouette': dbscan_silhouette,
        'inertia': dbscan_inertia
    })
    
    # 3. Contour-Based聚类
    print("\n" + "-" * 80)
    print("3. Contour-Based 聚类")
    print("-" * 80)
    contour_model, contour_labels, contour_inertia, contour_silhouette, contour_ami, contour_ari, contour_fmi = \
        analyzer.run_contour_clustering(
            grid_size=CONTOUR_PARAMS['grid_size'],
            max_buffer_distance=CONTOUR_PARAMS['max_buffer_distance']
        )
    
    n_clusters_found = len(set(contour_labels)) - (1 if -1 in contour_labels else 0)
    n_noise = list(contour_labels).count(-1)
    print(f"发现簇数: {n_clusters_found}, 噪声点: {n_noise}")
    print(f"Inertia:    {contour_inertia:.4f}")
    print(f"Silhouette: {contour_silhouette:.4f}")
    print(f"AMI:        {contour_ami:.4f}")
    print(f"ARI:        {contour_ari:.4f}")
    print(f"FMI:        {contour_fmi:.4f}")
    
    results.append({
        'name': 'Contour-Based',
        'labels': contour_labels,
        'ami': contour_ami,
        'ari': contour_ari,
        'fmi': contour_fmi,
        'silhouette': contour_silhouette,
        'inertia': contour_inertia
    })
    
    # 4. MATLAB Algorithm_PR（从保存的结果加载）
    if INCLUDE_MATLAB:
        print("\n" + "-" * 80)
        print("4. MATLAB Algorithm_PR（从保存的结果加载）")
        print("-" * 80)
        
        try:
            matlab_result = get_matlab_result(DATASET)
            matlab_labels = matlab_result['labels']
            
            # 计算额外指标
            matlab_inertia = analyzer.calculate_inertia(X, matlab_labels)
            
            if len(set(matlab_labels)) > 1:
                matlab_silhouette = silhouette_score(X, matlab_labels)
            else:
                matlab_silhouette = -1
            
            n_clusters_found = len(set(matlab_labels))
            print(f"发现簇数: {n_clusters_found}")
            print(f"Inertia:    {matlab_inertia:.4f}")
            print(f"Silhouette: {matlab_silhouette:.4f}")
            print(f"AMI:        {matlab_result['AMI']:.4f}")
            print(f"ARI:        {matlab_result['ARI']:.4f}")
            print(f"FMI:        {matlab_result['FMI']:.4f}")
            
            results.append({
                'name': 'MATLAB Algorithm_PR',
                'labels': matlab_labels,
                'ami': matlab_result['AMI'],
                'ari': matlab_result['ARI'],
                'fmi': matlab_result['FMI'],
                'silhouette': matlab_silhouette,
                'inertia': matlab_inertia
            })
            
        except (FileNotFoundError, KeyError) as e:
            print(f"无法加载MATLAB结果: {e}")
            print("请确保:")
            print("1. 已在MATLAB中运行 compare/algorithm/run_and_save_results.m")
            print(f"2. 数据集 '{DATASET}' 已被处理")
    
    # 汇总对比
    print("\n" + "=" * 90)
    print("算法对比汇总")
    print("=" * 90)
    print(f"{'算法':<25} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'Silhouette':<12} {'Inertia':<12}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['name']:<25} {r['ami']:<10.4f} {r['ari']:<10.4f} {r['fmi']:<10.4f} "
              f"{r['silhouette']:<12.4f} {r['inertia']:<12.2f}")
    
    print("=" * 90)
    
    # 可视化对比
    n_plots = len(results) + 1
    fig = plt.figure(figsize=(4 * n_plots, 4))
    
    # 真实标签
    plt.subplot(1, n_plots, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=20)
    plt.title('Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 各算法结果
    for i, r in enumerate(results, 2):
        plt.subplot(1, n_plots, i)
        plt.scatter(X[:, 0], X[:, 1], c=r['labels'], cmap='tab10', s=20)
        plt.title(f"{r['name']}\nARI={r['ari']:.3f}")
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n提示: 如果需要更新MATLAB结果，请在MATLAB中重新运行:")
    print("  compare/algorithm/run_and_save_results.m")
