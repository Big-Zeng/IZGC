import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from sklearn.neighbors import KernelDensity
import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics

# ==================== 配置区域 ====================
DATASET = "data/s1.csv"                # 数据集路径
NORMALIZE_COORDS = True                 # 是否归一化坐标到[0,100]范围
# ==================================================

# 读取 CSV 文件
df = pd.read_csv(DATASET)

# 提取 X, Y, target 列
X = df[['x', 'y']].astype(float).values  # 用于聚类的二维数据点
y_true = df['target'].values  # 实际标签


def normalize_coordinates(X):
    """
    归一化坐标到[0, 100]范围
    
    参数:
        X: 坐标数组 (n_samples, 2)
        
    返回:
        X_normalized: 归一化后的坐标
        scale_info: 缩放信息字典
    """
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 归一化到[0, 100]
    X_normalized = np.zeros_like(X)
    X_normalized[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 0
    X_normalized[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 0
    
    scale_info = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'x_range': x_range,
        'y_range': y_range
    }
    
    return X_normalized, scale_info


class VRPAnalyzer:
    def __init__(self, coords):
        self.coords = coords
        self.customer_points = [{'x': x, 'y': y} for x, y in coords]

    def run_kmeans_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = self.calculate_inertia(X, labels)
       # inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(X, labels)
        ari = adjusted_rand_score(y_true, labels)
        return kmeans, labels, inertia, silhouette_avg, ari

    def run_dbscan_clustering(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(X)

        # 计算轮廓系数和ARI（注意DBSCAN可能会有噪声点，标签为-1）
        if len(set(labels)) > 1:  # 确保有至少两个簇
            silhouette_avg = silhouette_score(X, labels)
            ari = adjusted_rand_score(y_true, labels)
        else:
            silhouette_avg = -1
            ari = -1

        inertia = self.calculate_inertia(X, labels)  # DBSCAN没有簇内误差平方和（Inertia）
        return dbscan, labels, inertia, silhouette_avg, ari

    def calculate_inertia(self, points, labels):
        """计算簇内误差平方和（Inertia）"""
        inertia = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_points = points[labels == label]
            if len(cluster_points) == 0:  # 如果该簇为空，跳过
                continue
            # 计算簇中心
            cluster_center = np.mean(cluster_points, axis=0)
            # 计算簇内所有点到簇中心的距离平方和
            inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
        return inertia

# grid_size是影响比较大的
     #ls.csv 2.36
    # max_buffer_distance先不用了
    def run_contour_clustering(self, grid_size=0.679, max_buffer_distance=0, min_cluster_size=2, radius=30, trueY=None):
        # 计算密度
        points = np.array(self.coords)
        tree = cKDTree(points)
        density = np.array([len(tree.query_ball_point(p, radius)) for p in points])  # zi = density
        # 计算 Voronoi
        # vor = Voronoi(points)
        #
        # # 初始化密度
        # density = np.zeros(len(points))
        #
        # # 对每个点计算自身 Voronoi 区域的面积
        # for i, region_index in enumerate(vor.point_region):
        #     region = vor.regions[region_index]
        #
        #     # region = [-1] 代表无穷大区域（边界点），跳过
        #     if -1 in region or len(region) == 0:
        #         density[i] = 0  # 可设为0 或 np.nan
        #         continue
        #
        #     polygon = vor.vertices[region]
        #
        #     # 使用多边形面积公式（shoelace formula）
        #     x = polygon[:, 0]
        #     y = polygon[:, 1]
        #     area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        #
        #     if area > 0:
        #         density[i] = 1.0 / area
        #     else:
        #         density[i] = 0
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]

        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance,
            min_cluster_size=5

        )

        #运行代码
        final_points = cluster.perform_clustering() #最终聚类的点的结果

        cluster.visualize_clusters()

        pre_labels = cluster.get_predicted_labels() # 预测的标签结果

        # 聚类（假设 ContourBasedClusteringWithoutZ 是你自定义的聚类算法）
        # 这里我们假设调用它的结果类似KMeans或DBSCAN的返回
        # 这里只是一个假设的返回，实际根据你的 `ContourBasedClusteringWithoutZ` 类进行修改
        if len(set(pre_labels)) >1:
            silhouette_avg = silhouette_score(X, pre_labels)
        else:
            silhouette_avg = -1

        if trueY is not None:
            ari = adjusted_rand_score(trueY, pre_labels)
        else:
            ari = adjusted_rand_score(y_true, pre_labels)

        inertia = self.calculate_inertia(X, pre_labels)  # 如果Contour算法不计算Inertia，可以设置为None
        metrics = ClusteringMetrics.evaluate_all(y_true, pre_labels)
        return final_points, pre_labels, inertia, silhouette_avg, ari


# 主程序执行部分
if __name__ == "__main__":
    print("="*80)
    print("聚类算法对比测试")
    print("="*80)
    
    # 显示原始数据信息
    print(f"\n数据集: {DATASET}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    print(f"\n原始数据范围:")
    print(f"  X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # 归一化坐标
    X_original = X.copy()
    scale_info = None
    
    if NORMALIZE_COORDS:
        print(f"\n正在归一化坐标...")
        X, scale_info = normalize_coordinates(X)
        print(f"✓ 坐标已归一化到[0, 100]范围")
        print(f"  归一化后X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
        print(f"  归一化后Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
        print(f"  缩放信息: X范围={scale_info['x_range']:.2f}, Y范围={scale_info['y_range']:.2f}")
    
    # 实例化VRPAnalyzer
    analyzer = VRPAnalyzer(X)
    
    print("\n" + "="*80)
    print("开始聚类测试...")
    print("="*80)


    # 1. 运行 KMeans 聚类
    print("\n1. KMeans聚类...")
    n_clusters = len(np.unique(y_true))  # 使用真实簇数
    kmeans_model, kmeans_labels, kmeans_inertia, kmeans_silhouette, kmeans_ari = analyzer.run_kmeans_clustering(
        n_clusters)

    # 2. 运行 DBSCAN 聚类
    print("2. DBSCAN聚类...")
    eps = 5.0 if NORMALIZE_COORDS else 5.0  # 根据归一化调整eps
    min_samples = 5
    dbscan_model, dbscan_labels, dbscan_inertia, dbscan_silhouette, dbscan_ari = analyzer.run_dbscan_clustering(eps=eps,
                                                                                                                min_samples=min_samples)

    # 3. 运行 Contour-based 聚类
    print("3. Contour-Based聚类...")

    
    contour_model, contour_labels, contour_inertia, contour_silhouette, contour_ari = analyzer.run_contour_clustering(

    )

    # 打印结果对比
    print("\n" + "="*80)
    print("聚类结果对比")
    print("="*80)
    print("\nKMeans Clustering Results:")
    print(f"  Inertia:    {kmeans_inertia:.4f}")
    print(f"  Silhouette: {kmeans_silhouette:.4f}")
    print(f"  ARI:        {kmeans_ari:.4f}")

    print("\nDBSCAN Clustering Results:")
    print(f"  Inertia:    {dbscan_inertia:.4f}")
    print(f"  Silhouette: {dbscan_silhouette:.4f}")
    print(f"  ARI:        {dbscan_ari:.4f}")

    print("\nContour-Based Clustering Results:")

    print(f"  Inertia:    {contour_inertia:.4f}")
    print(f"  Silhouette: {contour_silhouette:.4f}")
    print(f"  ARI:        {contour_ari:.4f}")
    print("="*80)

    # 可视化聚类结果（使用原始坐标以便更好地观察）
    X_plot = X_original if NORMALIZE_COORDS else X
    
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_true, cmap='viridis', s=30)
    plt.title('Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(142)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=kmeans_labels, cmap='viridis', s=30)
    plt.title(f'KMeans (ARI={kmeans_ari:.3f})')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(143)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=dbscan_labels, cmap='viridis', s=30)
    plt.title(f'DBSCAN (ARI={dbscan_ari:.3f})')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(144)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=contour_labels, cmap='viridis', s=30)
    plt.title(f'Contour-Based (ARI={contour_ari:.3f})')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存为: clustering_comparison.png")
    plt.show()
