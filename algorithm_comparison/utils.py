"""
Common utility functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys

# add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering_metrics import ClusteringMetrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, davies_bouldin_score


def load_dataset(filepath):
    """Load a dataset."""
    df = pd.read_csv(filepath)
    X = df[['x', 'y']].values.astype(float)
    # does the table carry a target column (ground truth)?
    if 'target' in df.columns:
        y_true = df['target'].values
    else:
        y_true = None  # unsupervised dataset
    return X, y_true


def load_real_dataset(filepath):
    """Load a real-world dataset without labels."""
    df = pd.read_csv(filepath)
    X = df[['x', 'y']].values.astype(float)
    return X


def normalize_coordinates(X):
    """Scale coordinates to [0, 100]."""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    X_norm = np.zeros_like(X)
    X_norm[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 50
    X_norm[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 50

    return X_norm


def calculate_inertia(X, labels):
    """Within-cluster sum of squares."""
    inertia = 0
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_points = X[labels == label]
        if len(cluster_points) == 0:
            continue
        center = np.mean(cluster_points, axis=0)
        inertia += np.sum(np.linalg.norm(cluster_points - center, axis=1) ** 2)
    return inertia


def calculate_disco_score(X, labels, eps=None):
    """
    Compute the DISCO index (Density-based Internal Score for Clusterings with nOise).
    
    DISCO extends the silhouette so that noise points are scored explicitly rather than ignored.
    
    Parameters:
        X: point coordinates, shape (n_samples, n_features)
        labels: cluster labels, -1 marks noise
        eps: density-connectivity radius, estimated automatically when None
    
    Returns:
        disco_score: DISCO value in [-1, 1], higher is better
        
    Reference:
        Beer et al. (2025). "Internal Evaluation of Density-Based Clusterings with Noise"
        arXiv:2503.00127
    """
    labels = np.array(labels)
    n_samples = len(labels)
    
    # edge cases
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters == 0:  # everything is noise
        return -1.0
    if n_clusters == 1 and -1 not in unique_labels:  # one cluster, no noise
        return 0.0
    
    # estimate eps when it is not supplied
    if eps is None:
        # median nearest-neighbour distance
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(5, n_samples)).fit(X)
        distances, _ = nbrs.kneighbors(X)
        eps = np.median(distances[:, -1])
    
    # pairwise distances, computed once
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(X, X, metric='euclidean')
    
    disco_scores = []
    
    for i in range(n_samples):
        label_i = labels[i]
        
        # score every point
        if label_i == -1:
            # noise points:
            #   not density-connected to any cluster -> correctly noise (+1)
            #   density-connected to some cluster -> should not be noise (-1)
            min_dist_to_cluster = float('inf')
            for cluster_label in unique_labels:
                if cluster_label == -1:
                    continue
                cluster_mask = labels == cluster_label
                cluster_points = X[cluster_mask]
                if len(cluster_points) > 0:
                    # distance to the nearest member of that cluster
                    distances_to_cluster = dist_matrix[i, cluster_mask]
                    min_dist_to_cluster = min(min_dist_to_cluster, np.min(distances_to_cluster))
            
            # noise scoring: far from every cluster (distance > eps) earns a positive score
            if min_dist_to_cluster > eps:
                disco_i = 1.0  # correct noise label
            else:
                disco_i = -1.0  # likely belongs to a cluster
                
        else:
            # cluster points: silhouette-like, using density connectivity
            same_cluster_mask = labels == label_i
            same_cluster_points = X[same_cluster_mask]
            
            if len(same_cluster_points) <= 1:
                disco_i = 0.0
            else:
                # a(i): mean distance to density-connected members of the same cluster
                distances_same = dist_matrix[i, same_cluster_mask]
                # density-connected members only (distance <= eps)
                density_connected_same = distances_same[distances_same <= eps]
                if len(density_connected_same) > 1:
                    a_i = np.mean(density_connected_same)
                else:
                    a_i = np.mean(distances_same)
                
                # b(i): mean distance to the nearest other cluster
                b_i = float('inf')
                for other_label in unique_labels:
                    if other_label == label_i or other_label == -1:
                        continue
                    other_cluster_mask = labels == other_label
                    if np.sum(other_cluster_mask) > 0:
                        distances_other = dist_matrix[i, other_cluster_mask]
                        avg_dist_other = np.mean(distances_other)
                        b_i = min(b_i, avg_dist_other)
                
                # silhouette-style score
                if b_i == float('inf'):
                    disco_i = 0.0
                else:
                    disco_i = (b_i - a_i) / max(a_i, b_i)
        
        disco_scores.append(disco_i)
    
    # mean DISCO over all points
    return np.mean(disco_scores)


def sigmoid_penalty_score(silhouette_score, outlier_ratio, beta=0.1, alpha=60):
    """
    Penalise the silhouette with a sigmoid of the noise ratio.

    Parameters:
    silhouette_score: raw silhouette S
    outlier_ratio: noise ratio R in [0, 1]
    beta: inflection point of the penalty, where the score is halved
    alpha: steepness; larger values approach a step function
    """
    # penalty factor in [0, 1]
    penalty_factor = 1 / (1 + np.exp(alpha * (outlier_ratio - beta)))

    return silhouette_score * penalty_factor

def evaluate_clustering(y_true, y_pred, X):
    """
    Evaluate a clustering result with supervised and internal indices.

    Weighted composite scheme used in this paper:
    - Silhouette_core: silhouette over non-noise points (cluster quality)
    - Outlier_ratio: fraction of noise points (coverage)
    - Composite: Silhouette_core * (1 - Outlier_ratio)
    """
    y_pred_array = np.array(y_pred)
    n_total = len(y_pred_array)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = int(np.sum(y_pred_array == -1))

    # core points (non-noise)
    mask_core = y_pred_array != -1
    n_core = int(np.sum(mask_core))
    outlier_ratio = n_noise / n_total if n_total > 0 else 1.0

    # silhouette over core points only
    silhouette_core = -1
    ch_score = 0
    dbi_score = float('inf')

    if n_clusters > 1 and n_core > 1:
        unique_core_labels = np.unique(y_pred_array[mask_core])
        if len(unique_core_labels) > 1:
            try:
                silhouette_core = silhouette_score(X[mask_core], y_pred_array[mask_core])
                ch_score = calinski_harabasz_score(X[mask_core], y_pred_array[mask_core])
                dbi_score = davies_bouldin_score(X[mask_core], y_pred_array[mask_core])
            except:
                pass

    # composite score (normalised weighted scheme)
    # shift the silhouette from [-1, 1] to [0, 1] so that the score stays positive
    # Score = ((Silhouette_core + 1) / 2) × (1 - Outlier_Ratio)^α
    # exponent alpha:
    #   - alpha = 1: linear penalty
    #   - alpha = 2: quadratic penalty (recommended), curbs discarding points to inflate the score
    #                dropping 20% of the data multiplies the score by 0.8^2 = 0.64
    alpha = 0.5  # penalty exponent (quadratic)
    if silhouette_core > -1:
        # normalise to [0, 1]
       # silhouette_normalized = (silhouette_core + 1) / 2
        # apply the noise penalty
        outlier_penalty = (1 - outlier_ratio) ** alpha
        # composite_score =  silhouette_core * outlier_penalty
        composite_score = sigmoid_penalty_score(silhouette_core, outlier_ratio)
    else:
        composite_score = -1

    # DISCO: density-aware index that scores noise explicitly
    disco_score = -1
    try:
        if n_clusters >= 1 and n_total > 1:
            disco_score = calculate_disco_score(X, y_pred_array)
    except Exception as e:
        # keep the default -1 when DISCO fails
        pass

    result = {
        'Silhouette_core': silhouette_core,  # core-point silhouette
        'Silhouette': silhouette_core,       # alias kept for backward compatibility
        'Outlier_ratio': outlier_ratio,      # noise fraction
        'Composite': composite_score,        # composite score (primary objective)
        'DISCO': disco_score,                # DISCO index (density-aware)
        'CH': ch_score,                      # Calinski-Harabasz index
        'DBI': dbi_score,                    # Davies-Bouldin index (lower is better)
        'Inertia': calculate_inertia(X, y_pred_array),
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

    # supervised indices, only when ground truth is available
    if y_true is not None:
        metrics = ClusteringMetrics.evaluate_all(y_true, y_pred)
        result['AMI'] = metrics['AMI']
        result['ARI'] = metrics['ARI']
        result['FMI'] = metrics['FMI']
    else:
        result['AMI'] = None
        result['ARI'] = None
        result['FMI'] = None

    return result


def evaluate_clustering_unsupervised(y_pred, X):
    """Evaluate a clustering result with internal indices only."""
    return evaluate_clustering(None, y_pred, X)


def plot_clustering_result(X, labels, title, save_path, figsize=(8, 6), dpi=600,
                           marker_size=30, show_legend=False, show_title=False):
    """
    Plot a clustering result at publication quality.

    Parameters:
        X: coordinate array
        labels: cluster labels
        title: figure title, drawn only when show_title is True
        save_path: output path
        figsize: figure size
        dpi: resolution (300 dpi for publication)
        marker_size: point size
        show_legend: draw the legend (default False)
        show_title: draw the title (default False)
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # font setup
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 16

    # fixed palette of 20 colours
    FIXED_COLORS = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # grey
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aec7e8',  # light blue
        '#ffbb78',  # light orange
        '#98df8a',  # light green
        '#ff9896',  # light red
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d2',  # light pink
        '#c7c7c7',  # light grey
        '#dbdb8d',  # light olive
        '#9edae5',  # light cyan
    ]

    # unique labels, sorted for reproducibility
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # draw each cluster
    for label in unique_labels:
        mask = np.array(labels) == label
        if label == -1:
            # noise points as grey crosses
            plt.scatter(X[mask, 0], X[mask, 1], c='gray', s=marker_size*2,
                       alpha=0.5, label='Noise', marker='x')
        else:
            # cycle through the fixed palette
            color = FIXED_COLORS[label % len(FIXED_COLORS)]
            plt.scatter(X[mask, 0], X[mask, 1], c=color,
                       s=marker_size, alpha=0.7, label=f'Cluster {label+1}',
                       edgecolors='white', linewidths=0.5)

    plt.xlabel('X',fontsize =20)
    plt.ylabel('Y',fontsize =20)
    plt.tick_params(axis="both", labelsize=20, length=3)
    if show_title:
        plt.title(title)

    if show_legend and n_clusters:
       plt.legend(loc='best', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Figure saved: {save_path}")

def plot_ground_truth(X, y_true, save_path, figsize=(8, 6), dpi=300, marker_size=30):
    """Plot the ground-truth labels."""
    plot_clustering_result(X, y_true, 'Ground Truth', save_path,
                          figsize=figsize, dpi=dpi, marker_size=marker_size)


def print_best_result(algorithm_name, best_params, best_metrics):
    """Print the best result."""
    print(f"\n{'='*60}")
    print(f"{algorithm_name}: best result")
    print(f"{'='*60}")
    print(f"Best parameters: {best_params}")

    # supervised indices
    if best_metrics.get('ARI') is not None:
        print(f"AMI: {best_metrics['AMI']:.4f}")
        print(f"ARI: {best_metrics['ARI']:.4f}")
        print(f"FMI: {best_metrics['FMI']:.4f}")

    # internal indices (composite scheme)
    if 'Composite' in best_metrics:
        print(f"Composite: {best_metrics['Composite']:.4f} (primary metric)")
        print(f"  ├─ Silhouette_core: {best_metrics['Silhouette_core']:.4f}")
        print(f"  └─ Outlier_ratio: {best_metrics['Outlier_ratio']:.4f}")
    elif 'Silhouette' in best_metrics:
        print(f"Silhouette: {best_metrics['Silhouette']:.4f}")

    print(f"CH: {best_metrics['CH']:.2f}")
    if 'DISCO' in best_metrics and best_metrics['DISCO'] != -1:
        print(f"DISCO: {best_metrics['DISCO']:.4f} (density-aware)")
    if 'DBI' in best_metrics:
        dbi_val = best_metrics['DBI']
        if dbi_val == float('inf'):
            print(f"DBI: inf")
        else:
            print(f"DBI: {dbi_val:.4f} (lower is better)")
    print(f"Clusters: {best_metrics['n_clusters']}, noise points: {best_metrics['n_noise']}")
    print(f"{'='*60}")


def save_best_result(algorithm_name, dataset_name, best_params, best_metrics, save_path):
    """
    Save the best result to a text file.

    Parameters:
        algorithm_name: algorithm name
        dataset_name: dataset name
        best_params: dictionary of the best parameters
        best_metrics: dictionary of the best metrics
        save_path: output path
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"{'='*60}\n\n")

        f.write("Best parameters:\n")
        for key, value in best_params.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")

        f.write("\nEvaluation metrics:\n")

        # supervised indices
        if best_metrics.get('ARI') is not None:
            f.write(f"  AMI: {best_metrics['AMI']:.4f}\n")
            f.write(f"  ARI: {best_metrics['ARI']:.4f}\n")
            f.write(f"  FMI: {best_metrics['FMI']:.4f}\n")

        # internal indices
        if 'Composite' in best_metrics:
            f.write(f"  Composite: {best_metrics['Composite']:.4f} (primary metric)\n")
            f.write(f"    ├─ Silhouette_core: {best_metrics['Silhouette_core']:.4f}\n")
            f.write(f"    └─ Outlier_ratio: {best_metrics['Outlier_ratio']:.4f}\n")
        elif 'Silhouette' in best_metrics:
            f.write(f"  Silhouette: {best_metrics['Silhouette']:.4f}\n")

        f.write(f"  CH: {best_metrics['CH']:.2f}\n")
        if 'DISCO' in best_metrics and best_metrics['DISCO'] != -1:
            f.write(f"  DISCO: {best_metrics['DISCO']:.4f} (density-aware)\n")
        if 'DBI' in best_metrics:
            dbi_val = best_metrics['DBI']
            if dbi_val == float('inf'):
                f.write(f"  DBI: inf\n")
            else:
                f.write(f"  DBI: {dbi_val:.4f} (lower is better)\n")
        f.write(f"  Clusters: {best_metrics['n_clusters']}\n")
        f.write(f"  Noise points: {best_metrics['n_noise']}\n")
        f.write(f"{'='*60}\n")
    
    print(f"  Result saved: {save_path}")
