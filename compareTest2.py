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

# ==================== Configuration ====================
DATASET = "data/flame.csv"                # Dataset path
NORMALIZE_COORDS = True                 # Whether to normalise coordinates to [0, 100]
# ==================================================

# Read the CSV file
df = pd.read_csv(DATASET)

# Extract the X, Y and target columns
X = df[['x', 'y']].astype(float).values  # 2-D points used for clustering
y_true = df['target'].values  # True labels


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
    
    # Normalise to [0, 100]
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

        # Compute the silhouette coefficient and ARI (DBSCAN may label noise as -1)
        if len(set(labels)) > 1:  # Ensure at least two clusters
            silhouette_avg = silhouette_score(X, labels)
            ari = adjusted_rand_score(y_true, labels)
        else:
            silhouette_avg = -1
            ari = -1

        inertia = self.calculate_inertia(X, labels)  # DBSCAN has no within-cluster sum of squares (inertia)
        return dbscan, labels, inertia, silhouette_avg, ari

    def calculate_inertia(self, points, labels):
        """计算簇内误差平方和（Inertia）"""
        inertia = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_points = points[labels == label]
            if len(cluster_points) == 0:  # Skip the cluster if it is empty
                continue
            # Compute the cluster centres
            cluster_center = np.mean(cluster_points, axis=0)
            # Sum of squared distances from all cluster points to the cluster centre
            inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
        return inertia

# grid_size has the largest effect
     #ls.csv 2.36
    # max_buffer_distance is currently unused
    def run_contour_clustering(self, grid_size=3.160 , max_buffer_distance=0, min_cluster_size=2, radius=30, trueY=None):
        # Compute density
        points = np.array(self.coords)
        tree = cKDTree(points)
        density = np.array([len(tree.query_ball_point(p, radius)) for p in points])  # zi = density
        # Compute the Voronoi diagram
        # vor = Voronoi(points)
        #
        # Initialise the density array
        # density = np.zeros(len(points))
        #
        # Compute the Voronoi cell area of every point
        # for i, region_index in enumerate(vor.point_region):
        #     region = vor.regions[region_index]
        #
        # region = [-1] marks the unbounded outer region (boundary points); skip it
        #     if -1 in region or len(region) == 0:
        # density[i] = 0  # 0 or np.nan are both acceptable
        #         continue
        #
        #     polygon = vor.vertices[region]
        #
        # Use the shoelace formula for the polygon area
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
            min_cluster_size=5,
            merge_small_clusters=True

        )

        #Run the code
        final_points = cluster.perform_clustering() #Final clustering result of the points

        cluster.visualize_clusters()

        pre_labels = cluster.get_predicted_labels() # Predicted labels

        # Cluster (ContourBasedClusteringWithoutZ is the custom algorithm used here)
        # Assume the call returns something like a K-means or DBSCAN result
        # Placeholder return value; adapt it to the ContourBasedClusteringWithoutZ class
        if len(set(pre_labels)) >1:
            silhouette_avg = silhouette_score(X, pre_labels)
        else:
            silhouette_avg = -1

        if trueY is not None:
            ari = adjusted_rand_score(trueY, pre_labels)
        else:
            ari = adjusted_rand_score(y_true, pre_labels)

        inertia = self.calculate_inertia(X, pre_labels)  # None is allowed when the contour algorithm does not compute inertia
        metrics = ClusteringMetrics.evaluate_all(y_true, pre_labels)
        return final_points, pre_labels, inertia, silhouette_avg, ari


# Entry point
if __name__ == "__main__":
    print("="*80)
    print("聚类算法对比测试")
    print("="*80)
    
    # Show the raw-data summary
    print(f"\n数据集: {DATASET}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    print(f"\n原始数据范围:")
    print(f"  X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # Normalised coordinates
    X_original = X.copy()
    scale_info = None
    
    if NORMALIZE_COORDS:
        print(f"\n正在归一化坐标...")
        X, scale_info = normalize_coordinates(X)
        print(f"✓ 坐标已归一化到[0, 100]范围")
        print(f"  归一化后X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
        print(f"  归一化后Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
        print(f"  缩放信息: X范围={scale_info['x_range']:.2f}, Y范围={scale_info['y_range']:.2f}")
    
    # Instantiate VRPAnalyzer
    analyzer = VRPAnalyzer(X)
    
    print("\n" + "="*80)
    print("开始聚类测试...")
    print("="*80)


    # 1. Run K-means clustering
    print("\n1. KMeans聚类...")
    n_clusters = len(np.unique(y_true))  # Use the true number of clusters
    kmeans_model, kmeans_labels, kmeans_inertia, kmeans_silhouette, kmeans_ari = analyzer.run_kmeans_clustering(
        n_clusters)

    # 2. Run DBSCAN clustering
    print("2. DBSCAN聚类...")
    eps = 5.0 if NORMALIZE_COORDS else 5.0  # Scale eps with the normalisation
    min_samples = 5
    dbscan_model, dbscan_labels, dbscan_inertia, dbscan_silhouette, dbscan_ari = analyzer.run_dbscan_clustering(eps=eps,
                                                                                                                min_samples=min_samples)

    # 3. Run contour-based clustering
    print("3. Contour-Based聚类...")

    
    contour_model, contour_labels, contour_inertia, contour_silhouette, contour_ari = analyzer.run_contour_clustering(

    )

    # Print the comparison
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

    # Visualise the clustering with the original coordinates for clarity
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
