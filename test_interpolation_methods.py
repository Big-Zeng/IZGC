"""
测试不同插值方法的聚类效果
对比 linear、cubic、idw 三种插值方法
"""

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

import ContourBasedClusteringWithoutZ4 as ContourClustering
from clustering_metrics import ClusteringMetrics


# Configuration
DATASET = "data/jain.csv"
GRID_SIZE = 0.617  # Set None to compute it automatically
MAX_BUFFER_DISTANCE = 5

# Interpolation method settings
INTERPOLATION_METHODS = {
    'linear': {
        'params': {}
    },
    'cubic': {
        'params': {
            'cubic_smoothing': 0.5  # Smoothing parameter
        }
    },
    'idw': {
        'params': {
            'idw_power': 2.0,       # Power
            'idw_neighbors': 8      # Number of neighbours
        }
    }
}


def prepare_data(filepath):
    """准备数据"""
    df = pd.read_csv(filepath)
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    # Compute density with Voronoi
    vor = Voronoi(X)
    density = np.zeros(len(X))
    
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
    
    # Prepare combined_data
    combined_data = [(*point, density[i], i) for i, point in enumerate(X)]
    
    return X, y_true, combined_data


def test_interpolation_method(method_name, combined_data, y_true, grid_size, params):
    """测试单个插值方法"""
    print(f"\n{'='*80}")
    print(f"测试插值方法: {method_name}")
    print(f"{'='*80}")
    print(f"参数: {params}")
    
    # Create clustering instance
    cluster = ContourClustering.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=grid_size,
        max_buffer_distance=MAX_BUFFER_DISTANCE,
        interpolation_method=method_name,
        **params
    )
    
    # Run clustering
    cluster.perform_clustering()
    
    # Get the predicted labels
    pred_labels = cluster.get_predicted_labels()
    
    # Compute evaluation metrics
    metrics = ClusteringMetrics.evaluate_all(y_true, pred_labels)
    
    # Compute the silhouette coefficient and inertia
    silhouette, inertia = cluster.silhouette_score()
    
    # Visualize results
    cluster.visualize_clusters()
    
    # Statistics
    n_clusters = len(cluster.density_centers)
    n_noise = list(pred_labels).count(-1)
    
    print(f"\n结果:")
    print(f"  发现簇数: {n_clusters}")
    print(f"  噪声点数: {n_noise}")
    print(f"  AMI:        {metrics['AMI']:.4f}")
    print(f"  ARI:        {metrics['ARI']:.4f}")
    print(f"  FMI:        {metrics['FMI']:.4f}")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Inertia:    {inertia:.2f}")
    
    return {
        'method': method_name,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'AMI': metrics['AMI'],
        'ARI': metrics['ARI'],
        'FMI': metrics['FMI'],
        'Silhouette': silhouette,
        'Inertia': inertia,
        'labels': pred_labels
    }


def main():
    print("="*80)
    print("插值方法对比测试")
    print("="*80)
    
    # Read data
    print(f"\n数据集: {DATASET}")
    df = pd.read_csv(DATASET)
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    # Prepare data
    X, y_true, combined_data = prepare_data(DATASET)
    
    # Test every interpolation method
    results = []
    for method_name, config in INTERPOLATION_METHODS.items():
        result = test_interpolation_method(
            method_name,
            combined_data,
            y_true,
            GRID_SIZE,
            config['params']
        )
        results.append(result)
    
    # Summary comparison
    print("\n" + "="*80)
    print("插值方法对比汇总")
    print("="*80)
    print(f"{'方法':<10} {'簇数':<8} {'噪声':<8} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'Silhouette':<12} {'Inertia':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['method']:<10} {r['n_clusters']:<8} {r['n_noise']:<8} "
              f"{r['AMI']:<10.4f} {r['ARI']:<10.4f} {r['FMI']:<10.4f} "
              f"{r['Silhouette']:<12.4f} {r['Inertia']:<12.2f}")
    
    print("="*80)
    
    # Find the best method
    best_ari = max(results, key=lambda x: x['ARI'])
    best_ami = max(results, key=lambda x: x['AMI'])
    
    print(f"\n最佳ARI: {best_ari['method']} (ARI={best_ari['ARI']:.4f})")
    print(f"最佳AMI: {best_ami['method']} (AMI={best_ami['AMI']:.4f})")
    
    # Visual comparison
    fig = plt.figure(figsize=(16, 4))
    
    # Ground-truth labels
    plt.subplot(1, len(results) + 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=20)
    plt.title('Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Results per method
    for i, r in enumerate(results, 2):
        plt.subplot(1, len(results) + 1, i)
        plt.scatter(X[:, 0], X[:, 1], c=r['labels'], cmap='tab10', s=20)
        plt.title(f"{r['method']}\nARI={r['ARI']:.3f}")
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
