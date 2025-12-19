"""
聚类算法对比测试脚本
对比 KMeans, DBSCAN 和 Contour-Based Clustering 三种算法
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 导入自定义模块
import ContourBasedClusteringWithoutZ3 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics


class ClusteringComparison:
    """聚类算法对比类"""
    
    def __init__(self, data_file, n_clusters=None):
        """
        初始化
        
        参数:
            data_file: CSV数据文件路径
            n_clusters: 真实的簇数量（如果已知）
        """
        # 读取数据
        df = pd.read_csv(data_file)
        
        # 提取坐标和标签
        if 'x' in df.columns and 'y' in df.columns:
            self.X = df[['x', 'y']].values
        else:
            # 如果没有x,y列，使用前两列
            self.X = df.iloc[:, :2].values
        
        # 提取真实标签
        if 'target' in df.columns:
            self.y_true = df['target'].values
        else:
            self.y_true = df.iloc[:, -1].values
        
        # 真实簇数
        self.n_clusters = n_clusters if n_clusters else len(np.unique(self.y_true))
        
        print(f"数据加载成功: {data_file}")
        print(f"样本数: {len(self.X)}, 真实簇数: {self.n_clusters}")
        print(f"X范围: [{self.X[:, 0].min():.2f}, {self.X[:, 0].max():.2f}]")
        print(f"Y范围: [{self.X[:, 1].min():.2f}, {self.X[:, 1].max():.2f}]")
    
    def calculate_inertia(self, points, labels):
        """计算簇内误差平方和（Inertia）"""
        inertia = 0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
            cluster_points = points[labels == label]
            if len(cluster_points) == 0:
                continue
            cluster_center = np.mean(cluster_points, axis=0)
            inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
        
        return inertia
    
    def run_kmeans(self):
        """运行KMeans聚类"""
        print("\n" + "=" * 70)
        print("运行 KMeans 聚类")
        print("=" * 70)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X)
        
        # 计算评估指标
        inertia = self.calculate_inertia(self.X, labels)
        
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(self.X, labels)
        else:
            silhouette = -1
        
        # 使用自定义评估指标
        metrics = ClusteringMetrics.evaluate_all(self.y_true, labels)
        
        results = {
            'algorithm': 'KMeans',
            'labels': labels,
            'inertia': inertia,
            'silhouette': silhouette,
            'ami': metrics['AMI'],
            'ari': metrics['ARI'],
            'fmi': metrics['FMI'],
            'accuracy': metrics.get('Accuracy', None)
        }
        
        self._print_results(results)
        return results
    
    def run_dbscan(self, eps=None, min_samples=5):
        """运行DBSCAN聚类"""
        print("\n" + "=" * 70)
        print("运行 DBSCAN 聚类")
        print("=" * 70)
        
        # 自动估计eps（如果未提供）
        if eps is None:
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(self.X)
            distances, indices = neighbors_fit.kneighbors(self.X)
            distances = np.sort(distances[:, -1], axis=0)
            eps = np.percentile(distances, 90)
            print(f"自动估计 eps = {eps:.4f}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(self.X)
        
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"发现簇数: {n_clusters_found}, 噪声点: {n_noise}")
        
        # 计算评估指标
        inertia = self.calculate_inertia(self.X, labels)
        
        if len(set(labels)) > 1 and n_clusters_found > 0:
            # 排除噪声点计算轮廓系数
            mask = labels != -1
            if np.sum(mask) > 0 and len(np.unique(labels[mask])) > 1:
                silhouette = silhouette_score(self.X[mask], labels[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        # 使用自定义评估指标
        metrics = ClusteringMetrics.evaluate_all(self.y_true, labels)
        
        results = {
            'algorithm': 'DBSCAN',
            'labels': labels,
            'inertia': inertia,
            'silhouette': silhouette,
            'ami': metrics['AMI'],
            'ari': metrics['ARI'],
            'fmi': metrics['FMI'],
            'accuracy': metrics.get('Accuracy', None),
            'n_clusters': n_clusters_found,
            'n_noise': n_noise
        }
        
        self._print_results(results)
        return results
    
    def run_contour_clustering(self, grid_size=1.0, max_buffer_distance=5):
        """运行等值线聚类"""
        print("\n" + "=" * 70)
        print("运行 Contour-Based 聚类")
        print("=" * 70)
        print(f"参数: grid_size={grid_size}, max_buffer_distance={max_buffer_distance}")
        
        # 计算密度（使用Voronoi）
        points = self.X
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
            
            if area > 0:
                density[i] = 1.0 / area
            else:
                density[i] = 0
        
        # 准备数据
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
        
        # 运行聚类
        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance
        )
        
        cluster.perform_clustering()
        labels = cluster.get_predicted_labels()
        
        # 可视化
        cluster.visualize_clusters()
        
        # 计算评估指标
        inertia = self.calculate_inertia(self.X, labels)
        
        if len(set(labels)) > 1:
            mask = np.array(labels) != -1
            if np.sum(mask) > 0 and len(np.unique(np.array(labels)[mask])) > 1:
                silhouette = silhouette_score(self.X[mask], np.array(labels)[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        # 使用自定义评估指标
        metrics = ClusteringMetrics.evaluate_all(self.y_true, labels)
        
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        results = {
            'algorithm': 'Contour-Based',
            'labels': labels,
            'inertia': inertia,
            'silhouette': silhouette,
            'ami': metrics['AMI'],
            'ari': metrics['ARI'],
            'fmi': metrics['FMI'],
            'accuracy': metrics.get('Accuracy', None),
            'n_clusters': n_clusters_found,
            'n_noise': n_noise
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results):
        """打印结果"""
        print(f"\n算法: {results['algorithm']}")
        print(f"  Inertia (簇内误差):     {results['inertia']:.4f}")
        print(f"  Silhouette (轮廓系数):  {results['silhouette']:.4f}")
        print(f"  AMI (调整互信息):       {results['ami']:.4f}")
        print(f"  ARI (调整兰德指数):     {results['ari']:.4f}")
        print(f"  FMI (F-Measure):       {results['fmi']:.4f}")
        if results.get('accuracy'):
            print(f"  Accuracy (准确率):      {results['accuracy']:.4f}")
        if 'n_clusters' in results:
            print(f"  发现簇数: {results['n_clusters']}, 噪声点: {results['n_noise']}")
    
    def compare_all(self, dbscan_eps=None, contour_grid_size=1.0):
        """对比所有算法"""
        print("\n" + "=" * 70)
        print("聚类算法对比测试")
        print("=" * 70)
        
        # 运行三种算法
        kmeans_results = self.run_kmeans()
        dbscan_results = self.run_dbscan(eps=dbscan_eps)
        contour_results = self.run_contour_clustering(grid_size=contour_grid_size)
        
        # 汇总对比
        self._print_comparison_table([kmeans_results, dbscan_results, contour_results])
        
        # 可视化对比
        self._visualize_comparison([kmeans_results, dbscan_results, contour_results])
        
        return kmeans_results, dbscan_results, contour_results
    
    def _print_comparison_table(self, results_list):
        """打印对比表格"""
        print("\n" + "=" * 90)
        print("算法对比汇总")
        print("=" * 90)
        print(f"{'算法':<20} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'Silhouette':<12} {'Inertia':<12}")
        print("-" * 90)
        
        for results in results_list:
            print(f"{results['algorithm']:<20} "
                  f"{results['ami']:<10.4f} "
                  f"{results['ari']:<10.4f} "
                  f"{results['fmi']:<10.4f} "
                  f"{results['silhouette']:<12.4f} "
                  f"{results['inertia']:<12.2f}")
        
        print("=" * 90)
    
    def _visualize_comparison(self, results_list):
        """可视化对比"""
        fig, axes = plt.subplots(1, len(results_list) + 1, figsize=(16, 4))
        
        # 绘制真实标签
        axes[0].scatter(self.X[:, 0], self.X[:, 1], c=self.y_true, cmap='tab10', s=20)
        axes[0].set_title('真实标签 (Ground Truth)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # 绘制各算法结果
        for i, results in enumerate(results_list, 1):
            axes[i].scatter(self.X[:, 0], self.X[:, 1], 
                          c=results['labels'], cmap='tab10', s=20)
            axes[i].set_title(f"{results['algorithm']}\nARI={results['ari']:.3f}")
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
        
        plt.tight_layout()
        plt.show()


# 主程序
if __name__ == "__main__":
    # 测试数据集
    dataset = "data/Aggregation.csv"  # 可以改为其他数据集
    
    # 创建对比实例
    comparison = ClusteringComparison(dataset)
    
    # 运行对比
    comparison.compare_all(
        dbscan_eps=1.5,           # DBSCAN的eps参数
        contour_grid_size=1.2     # Contour的grid_size参数
    )
