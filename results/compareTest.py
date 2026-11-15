import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from sklearn.neighbors import KernelDensity
import ContourBasedClusteringWithoutZ2 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics

# Read the CSV file
df = pd.read_csv("data/cth.csv")

# Extract the X, Y and target columns
X = df[['x', 'y']].values  # 2-D points used for clustering
y_true = df['target'].values  # True labels


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
    # max_buffer_distance is currently unused
    def run_contour_clustering(self, grid_size=4.24, max_buffer_distance=5, min_cluster_size=2, radius=1, trueY=None):
        # Compute density
        points = np.array(self.coords)
        # tree = cKDTree(points)
        # density = np.array([len(tree.query_ball_point(p, radius)) for p in points])  # zi = density
        # Compute the Voronoi diagram
        vor = Voronoi(points)

        # Initialise the density array
        density = np.zeros(len(points))

        # Compute the Voronoi cell area of every point
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]

            # region = [-1] marks the unbounded outer region (boundary points); skip it
            if -1 in region or len(region) == 0:
                density[i] = 0  # 0 or np.nan are both acceptable
                continue

            polygon = vor.vertices[region]

            # Use the shoelace formula for the polygon area
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
    # Instantiate VRPAnalyzer
    analyzer = VRPAnalyzer(X)
    x_mean = np.mean(X[:,0])
    y_mean =np.mean(X[:,1])
    print(f"Mean of x: {x_mean}")
    print(f"Mean of y: {y_mean}")
    x_max = np.max(X[:,0])
    x_min = np.min(X[:,0])
    y_max = np.min(X[:,1])

    #Normalise


    # 1. Run K-means clustering
    n_clusters = 3  # Set to the actual number of clusters
    kmeans_model, kmeans_labels, kmeans_inertia, kmeans_silhouette, kmeans_ari = analyzer.run_kmeans_clustering(
        n_clusters)

    # 2. Run DBSCAN clustering
    eps = 0.5  # Adjust to the actual data
    min_samples = 5
    dbscan_model, dbscan_labels, dbscan_inertia, dbscan_silhouette, dbscan_ari = analyzer.run_dbscan_clustering(eps=eps,
                                                                                                                min_samples=min_samples)

    # 3. Run contour-based clustering
    contour_model, contour_labels, contour_inertia, contour_silhouette, contour_ari = analyzer.run_contour_clustering()

    # Print the comparison
    print("KMeans Clustering Results:")
    print(f"Inertia: {kmeans_inertia}, Silhouette Score: {kmeans_silhouette}, ARI: {kmeans_ari}")

    print("\nDBSCAN Clustering Results:")
    print(f"Inertia: {dbscan_inertia}, Silhouette Score: {dbscan_silhouette}, ARI: {dbscan_ari}")

    print("\nContour-Based Clustering Results:")
    print(f"Inertia: {contour_inertia}, Silhouette Score: {contour_silhouette}, ARI: {contour_ari}")

    # Visualise the clustering results
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
