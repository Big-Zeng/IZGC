import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from sklearn.neighbors import KernelDensity
import ContourBasedClusteringWithoutZ3 as ContourBasedClusteringWithoutZ


# 读取 CSV 文件
df = pd.read_csv("data/Aggregation.csv")

# 提取 X, Y, target 列
X = df[['x', 'y']].values  # 用于聚类的二维数据点
y_true = df['target'].values  # 实际标签


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
    # max_buffer_distance先不用了
    def run_contour_clustering(self, grid_size=1.2, max_buffer_distance=2, min_cluster_size=2, radius=1, trueY=None):
        # 计算密度
        points = np.array(self.coords)
        # tree = cKDTree(points)
        # density = np.array([len(tree.query_ball_point(p, radius)) for p in points])  # zi = density
        # 计算 Voronoi
        vor = Voronoi(points)

        # 初始化密度
        density = np.zeros(len(points))

        # 对每个点计算自身 Voronoi 区域的面积
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]

            # region = [-1] 代表无穷大区域（边界点），跳过
            if -1 in region or len(region) == 0:
                density[i] = 0  # 可设为0 或 np.nan
                continue

            polygon = vor.vertices[region]

            # 使用多边形面积公式（shoelace formula）
            x = polygon[:, 0]
            y = polygon[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            if area > 0:
                density[i] = 1.0 / area
            else:
                density[i] = 0
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]

        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=max_buffer_distance,

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

        return final_points, pre_labels, inertia, silhouette_avg, ari


# 主程序执行部分
if __name__ == "__main__":
    # 实例化VRPAnalyzer
    analyzer = VRPAnalyzer(X)
    x_mean = np.mean(X[:,0])
    y_mean =np.mean(X[:,1])
    print(f"Mean of x: {x_mean}")
    print(f"Mean of y: {y_mean}")
    x_max = np.max(X[:,0])
    x_min = np.min(X[:,0])
    y_max = np.min(X[:,1])

    # 1. 运行 KMeans 聚类
    n_clusters = 3  # 设置为实际的簇数
    kmeans_model, kmeans_labels, kmeans_inertia, kmeans_silhouette, kmeans_ari = analyzer.run_kmeans_clustering(
        n_clusters)

    # 2. 运行 DBSCAN 聚类
    eps = 0.5  # 根据实际数据调整
    min_samples = 5
    dbscan_model, dbscan_labels, dbscan_inertia, dbscan_silhouette, dbscan_ari = analyzer.run_dbscan_clustering(eps=eps,
                                                                                                                min_samples=min_samples)

    # 3. 运行 Contour-based 聚类
    contour_model, contour_labels, contour_inertia, contour_silhouette, contour_ari = analyzer.run_contour_clustering()

    # 打印结果对比
    print("KMeans Clustering Results:")
    print(f"Inertia: {kmeans_inertia}, Silhouette Score: {kmeans_silhouette}, ARI: {kmeans_ari}")

    print("\nDBSCAN Clustering Results:")
    print(f"Inertia: {dbscan_inertia}, Silhouette Score: {dbscan_silhouette}, ARI: {dbscan_ari}")

    print("\nContour-Based Clustering Results:")
    print(f"Inertia: {contour_inertia}, Silhouette Score: {contour_silhouette}, ARI: {contour_ari}")

    # 可视化聚类结果
    plt.figure(figsize=(12, 8))

    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('KMeans Clustering')

    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title('DBSCAN Clustering')

    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=contour_labels, cmap='viridis')
    plt.title('Contour-Based Clustering')

    plt.show()
