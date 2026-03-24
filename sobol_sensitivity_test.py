"""
Sobol Sensitivity Analysis for ContourBasedClusteringWithoutZ22

Analyzes the impact of parameters on clustering metrics: ARI, FMI, AMI

Parameters:
- grid_size: Grid size for virtual point generation
- radius: Density calculation radius
- min_cluster_size: Minimum cluster size threshold
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, adjusted_mutual_info_score
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import warnings
import time
import os

import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DATASET = "data/2d_3c.csv"                # Dataset path
NORMALIZE_COORDS = True                   # Normalize coordinates to [0,100]
N_SAMPLES = 1000                            # Sobol samples (actual = N * (2D + 2))
OUTPUT_DIR = "sobol_results"              # Output directory
# =======================================================

# Define parameter problem
problem = {
    'num_vars': 3,
    'names': ['grid_size', 'radius', 'min_cluster_size'],
    'bounds': [
        [0.5, 5.0],      # grid_size: 0.5 to 5.0
        [5, 50],         # radius: density calculation radius 5 to 50
        [2, 20]          # min_cluster_size: 2 to 20
    ]
}


def normalize_coordinates(X):
    """Normalize coordinates to [0, 100] range"""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    X_normalized = np.zeros_like(X)
    X_normalized[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 0
    X_normalized[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 0
    
    return X_normalized


def run_contour_clustering(X, y_true, grid_size, radius, min_cluster_size):
    """
    Run contour-based clustering algorithm
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Calculate density
        points = np.array(X)
        tree = cKDTree(points)
        density = np.array([len(tree.query_ball_point(p, radius)) for p in points])
        
        # Prepare data
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
        
        # Create clusterer (max_buffer_distance fixed to 0)
        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=0,
            min_cluster_size=int(min_cluster_size),
            interactive=False,
            merge_small_clusters=True
        )
        
        # Perform clustering
        cluster.perform_clustering()
        
        # Get predicted labels
        pre_labels = cluster.get_predicted_labels()
        pre_labels = np.array(pre_labels)
        
        # Calculate metrics
        n_clusters = len(set(pre_labels)) - (1 if -1 in pre_labels else 0)
        n_noise = np.sum(pre_labels == -1)
        
        # If too few clusters, return default values
        if n_clusters < 2:
            return {
                'ari': 0,
                'fmi': 0,
                'ami': 0,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'success': False
            }
        
        # Calculate metrics
        ari = adjusted_rand_score(y_true, pre_labels)
        fmi = fowlkes_mallows_score(y_true, pre_labels)
        ami = adjusted_mutual_info_score(y_true, pre_labels)
        
        return {
            'ari': ari,
            'fmi': fmi,
            'ami': ami,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'success': True
        }
        
    except Exception as e:
        print(f"  Clustering failed: {e}")
        return {
            'ari': 0,
            'fmi': 0,
            'ami': 0,
            'n_clusters': 0,
            'n_noise': 0,
            'success': False
        }


def run_sobol_analysis(X, y_true, n_samples=64):
    """
    Run Sobol sensitivity analysis
    """
    print("\n" + "="*80)
    print("Sobol Sensitivity Analysis")
    print("="*80)
    
    # Generate Sobol samples
    print(f"\nGenerating Sobol samples (N={n_samples})...")
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
    print(f"Total samples: {len(param_values)}")
    
    # Store results
    Y_ari = np.zeros(len(param_values))
    Y_fmi = np.zeros(len(param_values))
    Y_ami = np.zeros(len(param_values))
    
    results_list = []
    
    # Run all parameter combinations
    print("\nRunning clustering tests...")
    start_time = time.time()
    
    for i, params in enumerate(param_values):
        grid_size, radius, min_cluster_size = params
        
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(param_values) - i - 1)
            print(f"  Progress: {i+1}/{len(param_values)} ({100*(i+1)/len(param_values):.1f}%) "
                  f"- Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        result = run_contour_clustering(
            X, y_true,
            grid_size=grid_size,
            radius=radius,
            min_cluster_size=min_cluster_size
        )
        
        Y_ari[i] = result['ari']
        Y_fmi[i] = result['fmi']
        Y_ami[i] = result['ami']
        
        results_list.append({
            'grid_size': grid_size,
            'radius': radius,
            'min_cluster_size': min_cluster_size,
            **result
        })
    
    total_time = time.time() - start_time
    print(f"\nCompleted! Total time: {total_time:.1f}s")
    
    # Save detailed results
    results_df = pd.DataFrame(results_list)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(f"{OUTPUT_DIR}/sobol_all_results.csv", index=False)
    print(f"\nDetailed results saved to: {OUTPUT_DIR}/sobol_all_results.csv")
    
    # Sobol analysis
    print("\n" + "-"*60)
    print("Sobol Sensitivity Index Analysis")
    print("-"*60)
    
    metrics = {
        'ARI': Y_ari,
        'FMI': Y_fmi,
        'AMI': Y_ami
    }
    
    sobol_results = {}
    
    for metric_name, Y in metrics.items():
        # Handle invalid values
        Y_clean = np.nan_to_num(Y, nan=0, posinf=0, neginf=0)
        
        # Check if there is enough variation
        if np.std(Y_clean) < 1e-10:
            print(f"\n{metric_name}: Data variation too small, skipping analysis")
            continue
        
        try:
            Si = sobol.analyze(problem, Y_clean, calc_second_order=True)
            sobol_results[metric_name] = Si
            
            print(f"\n{metric_name}:")
            print(f"  First-order indices (S1):")
            for j, name in enumerate(problem['names']):
                print(f"    {name}: {Si['S1'][j]:.4f} (conf: {Si['S1_conf'][j]:.4f})")
            
            print(f"  Total-order indices (ST):")
            for j, name in enumerate(problem['names']):
                print(f"    {name}: {Si['ST'][j]:.4f} (conf: {Si['ST_conf'][j]:.4f})")
                
        except Exception as e:
            print(f"\n{metric_name}: Analysis failed - {e}")
    
    return sobol_results, results_df


def create_sensitivity_table(sobol_results, output_dir):
    """Create sensitivity analysis summary table"""
    if not sobol_results:
        print("No valid Sobol results for table")
        return None
    
    # Build table data
    table_data = []
    param_names = problem['names']
    
    for metric_name, Si in sobol_results.items():
        for j, param in enumerate(param_names):
            table_data.append({
                'Metric': metric_name,
                'Parameter': param,
                'S1 (First-order)': Si['S1'][j],
                'S1_conf': Si['S1_conf'][j],
                'ST (Total-order)': Si['ST'][j],
                'ST_conf': Si['ST_conf'][j],
                'Interaction (ST-S1)': Si['ST'][j] - Si['S1'][j]
            })
    
    table_df = pd.DataFrame(table_data)
    
    # Save to CSV
    table_df.to_csv(f"{output_dir}/sensitivity_table.csv", index=False)
    print(f"\nSensitivity table saved to: {output_dir}/sensitivity_table.csv")
    
    # Create pivot table for better visualization
    pivot_s1 = table_df.pivot(index='Parameter', columns='Metric', values='S1 (First-order)')
    pivot_st = table_df.pivot(index='Parameter', columns='Metric', values='ST (Total-order)')
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY TABLE")
    print("="*80)
    
    print("\nFirst-order Sensitivity Indices (S1):")
    print("-" * 50)
    print(pivot_s1.to_string())
    
    print("\nTotal-order Sensitivity Indices (ST):")
    print("-" * 50)
    print(pivot_st.to_string())
    
    # Calculate parameter importance ranking
    print("\n" + "="*80)
    print("PARAMETER IMPORTANCE RANKING")
    print("="*80)
    
    avg_s1 = table_df.groupby('Parameter')['S1 (First-order)'].mean().sort_values(ascending=False)
    avg_st = table_df.groupby('Parameter')['ST (Total-order)'].mean().sort_values(ascending=False)
    
    print("\nBy Average S1 (Direct Effect):")
    for i, (param, val) in enumerate(avg_s1.items(), 1):
        print(f"  {i}. {param}: {val:.4f}")
    
    print("\nBy Average ST (Total Effect including interactions):")
    for i, (param, val) in enumerate(avg_st.items(), 1):
        print(f"  {i}. {param}: {val:.4f}")
    
    return table_df


def plot_sobol_results(sobol_results, output_dir):
    """Plot Sobol sensitivity analysis results (English)"""
    if not sobol_results:
        print("No valid Sobol results to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    param_names = problem['names']
    param_labels = ['grid_size', 'radius', 'min_cluster_size']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (metric_name, Si) in enumerate(sobol_results.items()):
        ax = axes[idx]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        # Plot S1 and ST
        bars1 = ax.bar(x - width/2, Si['S1'], width, label='S1 (First-order)', 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, Si['ST'], width, label='ST (Total-order)', 
                       color=colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')
        
        # Add error bars
        ax.errorbar(x - width/2, Si['S1'], yerr=Si['S1_conf'], fmt='none', 
                   color='black', capsize=3, linewidth=1)
        ax.errorbar(x + width/2, Si['ST'], yerr=Si['ST_conf'], fmt='none', 
                   color='black', capsize=3, linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars1, Si['S1']):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Sensitivity Index', fontsize=10)
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=15, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0, top=max(max(Si['ST']) * 1.3, 0.5))
    
    plt.suptitle('Sobol Sensitivity Analysis for Contour-Based Clustering', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sobol_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"\nSensitivity plot saved to: {output_dir}/sobol_sensitivity.png")
    plt.show()


def plot_parameter_effects(results_df, output_dir):
    """Plot parameter effects on metrics (English)"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    params = ['grid_size', 'radius', 'min_cluster_size']
    param_labels = ['Grid Size', 'Radius', 'Min Cluster Size']
    metrics = ['ari', 'fmi', 'ami']
    metric_labels = ['ARI', 'FMI', 'AMI']
    
    for i, (param, param_label) in enumerate(zip(params, param_labels)):
        for j, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]
            
            # Filter valid data
            valid_mask = results_df['success'] == True
            
            x = results_df.loc[valid_mask, param]
            y = results_df.loc[valid_mask, metric]
            
            ax.scatter(x, y, alpha=0.4, s=15, c='#3498db')
            
            # Add trend line
            if len(x) > 2:
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)
            
            ax.set_xlabel(param_label, fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.grid(alpha=0.3, linestyle='--')
            
            # Add correlation coefficient
            if len(x) > 2:
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Parameter Effects on Clustering Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_effects.png", dpi=300, bbox_inches='tight')
    print(f"Parameter effects plot saved to: {output_dir}/parameter_effects.png")
    plt.show()


def plot_heatmap_table(sobol_results, output_dir):
    """Create heatmap visualization of sensitivity indices"""
    if not sobol_results:
        return
    
    # Prepare data for heatmap
    metrics = list(sobol_results.keys())
    params = problem['names']
    
    s1_data = np.array([[sobol_results[m]['S1'][j] for m in metrics] for j in range(len(params))])
    st_data = np.array([[sobol_results[m]['ST'][j] for m in metrics] for j in range(len(params))])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # S1 heatmap
    im1 = axes[0].imshow(s1_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels(metrics)
    axes[0].set_yticks(range(len(params)))
    axes[0].set_yticklabels(params)
    axes[0].set_title('First-order Sensitivity (S1)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(params)):
        for j in range(len(metrics)):
            text = axes[0].text(j, i, f'{s1_data[i, j]:.3f}',
                               ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # ST heatmap
    im2 = axes[1].imshow(st_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels(metrics)
    axes[1].set_yticks(range(len(params)))
    axes[1].set_yticklabels(params)
    axes[1].set_title('Total-order Sensitivity (ST)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(params)):
        for j in range(len(metrics)):
            text = axes[1].text(j, i, f'{st_data[i, j]:.3f}',
                               ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.suptitle('Sobol Sensitivity Indices Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Sensitivity heatmap saved to: {output_dir}/sensitivity_heatmap.png")
    plt.show()


def find_best_parameters(results_df):
    """Find best parameter combinations"""
    print("\n" + "="*80)
    print("BEST PARAMETER COMBINATIONS")
    print("="*80)
    
    # Filter successful results
    valid_df = results_df[results_df['success'] == True].copy()
    
    if len(valid_df) == 0:
        print("No successful clustering results")
        return
    
    # Find best by different metrics
    metrics_info = [
        ('ari', 'ARI (Adjusted Rand Index)'),
        ('fmi', 'FMI (Fowlkes-Mallows Index)'),
        ('ami', 'AMI (Adjusted Mutual Information)')
    ]
    
    for metric, metric_name in metrics_info:
        best_idx = valid_df[metric].idxmax()
        best = valid_df.loc[best_idx]
        
        print(f"\nBest {metric_name}:")
        print(f"  grid_size: {best['grid_size']:.3f}")
        print(f"  radius: {best['radius']:.1f}")
        print(f"  min_cluster_size: {best['min_cluster_size']:.0f}")
        print(f"  {metric.upper()}: {best[metric]:.4f}")
        print(f"  Clusters: {best['n_clusters']:.0f}, Noise: {best['n_noise']:.0f}")


# Main program
if __name__ == "__main__":
    print("="*80)
    print("ContourBasedClustering Sobol Sensitivity Analysis")
    print("="*80)
    
    # Load data
    print(f"\nLoading dataset: {DATASET}")
    df = pd.read_csv(DATASET)
    X = df[['x', 'y']].astype(float).values
    y_true = df['target'].values
    
    print(f"Samples: {len(X)}")
    print(f"True clusters: {len(np.unique(y_true))}")
    
    # Normalize
    if NORMALIZE_COORDS:
        print("\nNormalizing coordinates to [0, 100]...")
        X = normalize_coordinates(X)
    
    # Run Sobol analysis
    sobol_results, results_df = run_sobol_analysis(X, y_true, n_samples=N_SAMPLES)
    
    # Create sensitivity table
    table_df = create_sensitivity_table(sobol_results, OUTPUT_DIR)
    
    # Plot results
    plot_sobol_results(sobol_results, OUTPUT_DIR)
    plot_parameter_effects(results_df, OUTPUT_DIR)
    plot_heatmap_table(sobol_results, OUTPUT_DIR)
    
    # Find best parameters
    find_best_parameters(results_df)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("="*80)
