"""
Run the full algorithm comparison.
Optimises every parameter automatically and produces publication-quality figures.
"""

# disable the interactive backend to speed up rendering
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
import sys
import time
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering,
    plot_clustering_result, plot_ground_truth
)
from algorithm_comparison.alg_contour import run_contour_clustering
from algorithm_comparison.alg_kmeans import run_kmeans
from algorithm_comparison.alg_dbscan import run_dbscan
from algorithm_comparison.alg_clique2 import run_clique2
from algorithm_comparison.alg_gmm import run_gmm, COVARIANCE_TYPES
from algorithm_comparison.alg_bang import run_bang, PYCLUSTERING_AVAILABLE as BANG_AVAILABLE

# try to import the MATLAB engine
try:
    from algorithm_comparison.alg_dpc_fwsn import run_dpc_fwsn, MATLAB_AVAILABLE
except:
    MATLAB_AVAILABLE = False

# ==================== configuration ====================
DATASET = "../data/s1.csv"
OUTPUT_DIR = "results"
NORMALIZE = True
OPTIMIZE_METRIC = 'ARI'

# parameter search ranges per algorithm (start, end, step)
CONTOUR_RANGES = {
    'grid_size': (0.65, 0.7, 0.001),           # 0.65 -> 0.7, step 0.001
    'max_buffer_distance': (0, 0, 1),        # fixed at 0
    'min_cluster_size': (5, 5, 1),          # fixed at 5
    'density_radius': (30, 30, 1)            # fixed at 30
}

KMEANS_RANGES = {
    'n_clusters': (15, 15, 1),  # fixed at 15
    'n_init': (10, 50, 1),  # 10 -> 50, step 1
    'max_iter': (300, 500, 50)  # 300 -> 500, step 50
}

DBSCAN_RANGES = {
    'eps': (1.0, 50, 1.0),  # neighbourhood radius, 1.0 -> 50.0, step 1.0
    'min_samples': (2, 50, 1)  # 2 -> 50, step 1
}

CLIQUE_RANGES = {
    'intervals': (4, 100, 1),  # 4 -> 100, step 1
    'density_threshold': (1, 50, 1)  # 1 -> 50, step 1
}

GMM_RANGES = {
    'n_components': (15, 15, 1),    # fixed at 15
    'n_init': (10, 50, 1)           # 10 -> 50, step 1
}

DPC_RANGES = {
    'k': (1, 50, 1)                          # 1 -> 50, step 1
}

BANG_RANGES = {
    'levels': (4, 50, 1),                  # 4 -> 50, step 1
    'density_threshold': (0.01, 1, 0.05)    # 0.01 -> 1, step 0.05
}

RUN_MATLAB = True
# ==============================================


def generate_param_values(start, end, step):
    """Build the value list for one parameter."""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def optimize_algorithm(name, run_func, X, y_true, param_ranges, optimize_metric='ARI', **extra_args):
    """Generic parameter-optimisation routine."""
    print(f"\n{'='*60}")
    print(f"Optimising {name}")
    print(f"{'='*60}")
    
    # build the value grid
    param_grid = {}
    for pname, (start, end, step) in param_ranges.items():
        param_grid[pname] = generate_param_values(start, end, step)
        print(f"  {pname}: {start} -> {end}, step={step}, {len(param_grid[pname])} values")
    
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_metrics = None
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total = np.prod([len(v) for v in param_values])
    
    print(f"Parameter combinations: {total}")
    
    start_time = time.time()
    
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        try:
            labels = run_func(X, **params, **extra_args)
            if labels is None:
                continue
            
            metrics = evaluate_clustering(y_true, labels, X)
            
            if metrics['n_clusters'] < 2:
                continue
            
            score = metrics[optimize_metric]
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_labels = labels.copy() if isinstance(labels, np.ndarray) else np.array(labels)
                best_metrics = metrics.copy()
                
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    
    if best_params:
        print(f"\nBest parameters: {best_params}")
        print(f"Best {optimize_metric}: {best_score:.4f}")
        print(f"Clusters: {best_metrics['n_clusters']}, noise: {best_metrics['n_noise']}")
        print(f"Elapsed: {elapsed:.2f}s")
    else:
        print(f"\nNo valid parameter combination found")
    
    return {
        'name': name,
        'params': best_params,
        'labels': best_labels,
        'metrics': best_metrics,
        'time': elapsed
    }


def main():
    """Entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("Clustering benchmark with automatic parameter optimisation")
    print("="*70)
    
    # load the data
    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Samples: {len(X)}")
    print(f"Ground-truth clusters: {len(np.unique(y_true))}")
    
    X_original = X.copy()
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("Coordinates scaled to [0, 100]")
    
    results = []
    
    # 1. Ground Truth
    print("\nSaving the ground-truth figure...")
    plot_ground_truth(X, y_true, f"{OUTPUT_DIR}/01_GroundTruth_{dataset_name}.png", dpi=300)
    
    # 2. ContourBasedClustering
    result = optimize_algorithm(
        "ContourBased", run_contour_clustering, X, y_true, 
        CONTOUR_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"ContourBased (ARI={result['metrics']['ARI']:.3f})",
            f"{OUTPUT_DIR}/02_ContourBased_{dataset_name}.png", dpi=300
        )
    
    # 3. K-means++
    result = optimize_algorithm(
        "K-means++", run_kmeans, X, y_true,
        KMEANS_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"K-means++ (ARI={result['metrics']['ARI']:.3f})",
            f"{OUTPUT_DIR}/03_KMeans_{dataset_name}.png", dpi=300
        )
    
    # 4. DBSCAN
    result = optimize_algorithm(
        "DBSCAN", run_dbscan, X, y_true,
        DBSCAN_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"DBSCAN (ARI={result['metrics']['ARI']:.3f})",
            f"{OUTPUT_DIR}/04_DBSCAN_{dataset_name}.png", dpi=300
        )
    
    # 5. CLIQUE
    result = optimize_algorithm(
        "CLIQUE", run_clique2, X, y_true,
        CLIQUE_RANGES, OPTIMIZE_METRIC
    )
    if result['labels'] is not None:
        results.append(result)
        plot_clustering_result(
            X, result['labels'],
            f"CLIQUE (ARI={result['metrics']['ARI']:.3f})",
            f"{OUTPUT_DIR}/05_CLIQUE_{dataset_name}.png", dpi=300
        )
    
    # 6. GMM (Gaussian mixture model)
    print(f"\n{'='*60}")
    print("Optimising GMM")
    print(f"{'='*60}")
    
    # GMM needs its own handling of the covariance_type parameter
    gmm_param_grid = {}
    for pname, (start, end, step) in GMM_RANGES.items():
        gmm_param_grid[pname] = generate_param_values(start, end, step)
        print(f"  {pname}: {start} -> {end}, step={step}, {len(gmm_param_grid[pname])} values")
    print(f"  covariance_type: {COVARIANCE_TYPES}")
    
    best_gmm_score = -np.inf
    best_gmm_params = None
    best_gmm_labels = None
    best_gmm_metrics = None
    
    total_gmm = len(gmm_param_grid['n_components']) * len(gmm_param_grid['n_init']) * len(COVARIANCE_TYPES)
    print(f"Parameter combinations: {total_gmm}")
    
    for n_components in gmm_param_grid['n_components']:
        for n_init in gmm_param_grid['n_init']:
            for cov_type in COVARIANCE_TYPES:
                try:
                    labels = run_gmm(X, n_components, cov_type, n_init)
                    if labels is None:
                        continue
                    metrics = evaluate_clustering(y_true, labels, X)
                    if metrics['n_clusters'] < 2:
                        continue
                    score = metrics[OPTIMIZE_METRIC]
                    if score > best_gmm_score:
                        best_gmm_score = score
                        best_gmm_params = {'n_components': n_components, 'covariance_type': cov_type, 'n_init': n_init}
                        best_gmm_labels = labels.copy()
                        best_gmm_metrics = metrics.copy()
                except:
                    pass
    
    if best_gmm_params:
        print(f"\nBest parameters: {best_gmm_params}")
        print(f"Best {OPTIMIZE_METRIC}: {best_gmm_score:.4f}")
        results.append({
            'name': 'GMM',
            'params': best_gmm_params,
            'labels': best_gmm_labels,
            'metrics': best_gmm_metrics,
            'time': 0
        })
        plot_clustering_result(
            X, best_gmm_labels,
            f"GMM (ARI={best_gmm_metrics['ARI']:.3f})",
            f"{OUTPUT_DIR}/06_GMM_{dataset_name}.png", dpi=300
        )
    
    # 7. BANG
    if BANG_AVAILABLE:
        result = optimize_algorithm(
            "BANG", run_bang, X, y_true,
            BANG_RANGES, OPTIMIZE_METRIC
        )
        if result['labels'] is not None:
            results.append(result)
            plot_clustering_result(
                X, result['labels'],
                f"BANG (ARI={result['metrics']['ARI']:.3f})",
                f"{OUTPUT_DIR}/07_BANG_{dataset_name}.png", dpi=300
            )
    
    # 8. DPC-FWSN (MATLAB)
    if RUN_MATLAB and MATLAB_AVAILABLE:
        print(f"\n{'='*60}")
        print("Optimising DPC-FWSN (MATLAB)")
        print(f"{'='*60}")
        
        # build the value grid
        k_values = generate_param_values(*DPC_RANGES['k'])
        print(f"  k: {DPC_RANGES['k'][0]} -> {DPC_RANGES['k'][1]}, step={DPC_RANGES['k'][2]}, {len(k_values)} values")
        
        try:
            from matlab_integration import MatlabRunner
            matlab_runner = MatlabRunner(method='auto')
            
            best_score = -np.inf
            best_params = None
            best_labels = None
            best_metrics = None
            
            n_clusters = len(np.unique(y_true))
            data_with_labels = np.column_stack([X, y_true])
            
            for k in k_values:
                try:
                    labels, metrics = matlab_runner.run_algorithm_pr(
                        data_with_labels, n_clusters, k
                    )
                    if labels is not None:
                        eval_metrics = evaluate_clustering(y_true, labels, X)
                        score = eval_metrics[OPTIMIZE_METRIC]
                        print(f"  k={k}: {OPTIMIZE_METRIC}={score:.4f}")
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'k': k}
                            best_labels = labels.copy()
                            best_metrics = eval_metrics.copy()
                except Exception as e:
                    print(f"  k={k}: failed - {e}")
            
            matlab_runner.close()
            
            if best_labels is not None:
                results.append({
                    'name': 'DPC-FWSN',
                    'params': best_params,
                    'labels': best_labels,
                    'metrics': best_metrics,
                    'time': 0
                })
                plot_clustering_result(
                    X, best_labels,
                    f"DPC-FWSN (ARI={best_metrics['ARI']:.3f})",
                    f"{OUTPUT_DIR}/08_DPC-FWSN_{dataset_name}.png", dpi=300
                )
        except Exception as e:
            print(f"DPC-FWSN failed: {e}")
    
    # summary
    print("\n" + "="*90)
    print("Algorithm comparison summary")
    print("="*90)
    
    header = f"{'Algorithm':<15} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'Silhouette':<12} {'Clusters':<10} {'Noise':<8}"
    print(header)
    print("-"*90)
    
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<15} {m['AMI']:<10.4f} {m['ARI']:<10.4f} {m['FMI']:<10.4f} "
              f"{m['Silhouette']:<12.4f} {m['n_clusters']:<8} {m['n_noise']:<8}")
    
    print("="*90)
    
    # write the CSV
    csv_data = []
    for r in results:
        m = r['metrics']
        csv_data.append({
            'Algorithm': r['name'],
            'Best_Params': str(r['params']),
            'AMI': m['AMI'],
            'ARI': m['ARI'],
            'FMI': m['FMI'],
            'Silhouette': m['Silhouette'],
            'n_clusters': m['n_clusters'],
            'n_noise': m['n_noise']
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f"{OUTPUT_DIR}/comparison_results_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    
    # dump the best result of each algorithm to a text file
    for r in results:
        txt_path = f"{OUTPUT_DIR}/{r['name']}_{dataset_name}_best.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Algorithm: {r['name']}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Best parameters: {r['params']}\n")
            f.write(f"\nEvaluation metrics:\n")
            f.write(f"  AMI: {r['metrics']['AMI']:.4f}\n")
            f.write(f"  ARI: {r['metrics']['ARI']:.4f}\n")
            f.write(f"  FMI: {r['metrics']['FMI']:.4f}\n")
            f.write(f"  Silhouette: {r['metrics']['Silhouette']:.4f}\n")
            f.write(f"  Clusters: {r['metrics']['n_clusters']}\n")
            f.write(f"  Noise points: {r['metrics']['n_noise']}\n")
        print(f"  Saved: {txt_path}")
    
    print(f"\nFigures written to: {OUTPUT_DIR}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
