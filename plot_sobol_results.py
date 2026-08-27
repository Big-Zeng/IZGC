"""
Plot Sobol sensitivity analysis results from saved data
Uses mathematical symbols: R (radius), Δg (grid_size), θ (min_cluster_size)
"""

import pandas as pd
import numpy as np
from SALib.analyze import sobol
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend first
import matplotlib.pyplot as plt
import os

# Enable LaTeX-style rendering
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# ==================== Configuration ====================
OUTPUT_DIR = "sobol_results"
RESULTS_FILE = f"{OUTPUT_DIR}/sobol_all_results.csv"
# =======================================================

# Parameter mapping: original name -> LaTeX symbol
PARAM_SYMBOLS = {
    'grid_size': r'$\Delta_g$',
    'radius': r'$R$', 
    'min_cluster_size': r'$\theta$'
}

# Define parameter problem (must match original analysis)
problem = {
    'num_vars': 3,
    'names': ['grid_size', 'radius', 'min_cluster_size'],
    'bounds': [
        [0.5, 5.0],
        [5, 50],
        [2, 20]
    ]
}


def load_results():
    """Load saved results"""
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Results file not found: {RESULTS_FILE}")
        return None
    
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} results from {RESULTS_FILE}")
    return df


def compute_sobol_indices(results_df):
    """Recompute Sobol indices from saved results"""
    metrics = {
        'ARI': results_df['ari'].values,
        'FMI': results_df['fmi'].values,
        'AMI': results_df['ami'].values
    }
    
    sobol_results = {}
    
    for metric_name, Y in metrics.items():
        Y_clean = np.nan_to_num(Y, nan=0, posinf=0, neginf=0)
        
        if np.std(Y_clean) < 1e-10:
            print(f"{metric_name}: Data variation too small, skipping")
            continue
        
        try:
            Si = sobol.analyze(problem, Y_clean, calc_second_order=True)
            sobol_results[metric_name] = Si
            print(f"{metric_name}: Sobol analysis completed")
        except Exception as e:
            print(f"{metric_name}: Analysis failed - {e}")
    
    return sobol_results


def plot_sobol_results(sobol_results, output_dir):
    """Plot Sobol sensitivity analysis with mathematical symbols"""
    if not sobol_results:
        print("No valid Sobol results to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    param_names = problem['names']
    param_labels = [PARAM_SYMBOLS[p] for p in param_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (metric_name, Si) in enumerate(sobol_results.items()):
        ax = axes[idx]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        # Plot S1 and ST
        bars1 = ax.bar(x - width/2, Si['S1'], width, label='$S_1$ (First-order)', 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, Si['ST'], width, label='$S_T$ (Total-order)', 
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
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Sensitivity Index', fontsize=11)
        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0, top=max(max(Si['ST']) * 1.3, 0.5))
    
    # plt.suptitle('Sobol Sensitivity Analysis for Contour-Based Clustering',
    #              fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sobol_sensitivity_symbols.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/sobol_sensitivity_symbols.png")
    plt.close(fig)


def plot_parameter_effects(results_df, output_dir):
    """Plot parameter effects with mathematical symbols"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    params = ['grid_size', 'radius', 'min_cluster_size']
    param_labels = [PARAM_SYMBOLS[p] for p in params]
    metrics = ['ari', 'fmi', 'ami']
    metric_labels = ['ARI', 'FMI', 'AMI']
    
    for i, (param, param_label) in enumerate(zip(params, param_labels)):
        for j, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]
            
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
            
            ax.set_xlabel(param_label, fontsize=12)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            
            # Add correlation coefficient
            if len(x) > 2:
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # plt.suptitle('Parameter Effects on Clustering Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_effects_symbols.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/parameter_effects_symbols.png")
    plt.close(fig)


def plot_heatmap_table(sobol_results, output_dir):
    """Create heatmap with mathematical symbols"""
    if not sobol_results:
        return
    
    metrics = list(sobol_results.keys())
    params = problem['names']
    param_labels = [PARAM_SYMBOLS[p] for p in params]
    
    s1_data = np.array([[sobol_results[m]['S1'][j] for m in metrics] for j in range(len(params))])
    st_data = np.array([[sobol_results[m]['ST'][j] for m in metrics] for j in range(len(params))])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # S1 heatmap
    im1 = axes[0].imshow(s1_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels(metrics, fontsize=11)
    axes[0].set_yticks(range(len(params)))
    axes[0].set_yticklabels(param_labels, fontsize=12)
    axes[0].set_title('First-order Sensitivity ($S_1$)', fontweight='bold', fontsize=12)
    
    for i in range(len(params)):
        for j in range(len(metrics)):
            axes[0].text(j, i, f'{s1_data[i, j]:.3f}',
                        ha='center', va='center', color='black', fontsize=11)
    
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # ST heatmap
    im2 = axes[1].imshow(st_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels(metrics, fontsize=11)
    axes[1].set_yticks(range(len(params)))
    axes[1].set_yticklabels(param_labels, fontsize=12)
    axes[1].set_title('Total-order Sensitivity ($S_T$)', fontweight='bold', fontsize=12)
    
    for i in range(len(params)):
        for j in range(len(metrics)):
            axes[1].text(j, i, f'{st_data[i, j]:.3f}',
                        ha='center', va='center', color='black', fontsize=11)
    
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.suptitle('Sobol Sensitivity Indices Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_heatmap_symbols.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/sensitivity_heatmap_symbols.png")
    plt.close(fig)


def create_latex_table(sobol_results, output_dir):
    """Create LaTeX-formatted sensitivity table"""
    if not sobol_results:
        return
    
    metrics = list(sobol_results.keys())
    params = problem['names']
    
    # Build table
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS TABLE (LaTeX format)")
    print("="*80)
    
    # Header
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Sobol Sensitivity Indices}\n"
    latex += "\\begin{tabular}{l" + "cc" * len(metrics) + "}\n"
    latex += "\\hline\n"
    
    # Column headers
    header = "Parameter"
    for m in metrics:
        header += f" & \\multicolumn{{2}}{{c}}{{{m}}}"
    latex += header + " \\\\\n"
    
    subheader = ""
    for m in metrics:
        subheader += " & $S_1$ & $S_T$"
    latex += subheader + " \\\\\n"
    latex += "\\hline\n"
    
    # Data rows
    for j, param in enumerate(params):
        row = PARAM_SYMBOLS[param].replace('$', '')
        for m in metrics:
            s1 = sobol_results[m]['S1'][j]
            st = sobol_results[m]['ST'][j]
            row += f" & {s1:.3f} & {st:.3f}"
        latex += "$" + row + "$ \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    print(latex)
    
    # Save to file
    with open(f"{output_dir}/sensitivity_table.tex", 'w') as f:
        f.write(latex)
    print(f"\nSaved: {output_dir}/sensitivity_table.tex")
    
    # Also create simple CSV with symbols
    table_data = []
    for j, param in enumerate(params):
        row = {'Parameter': PARAM_SYMBOLS[param]}
        for m in metrics:
            row[f'{m}_S1'] = sobol_results[m]['S1'][j]
            row[f'{m}_ST'] = sobol_results[m]['ST'][j]
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(f"{output_dir}/sensitivity_table_symbols.csv", index=False)
    print(f"Saved: {output_dir}/sensitivity_table_symbols.csv")


# Main
if __name__ == "__main__":
    print("="*80)
    print("Plotting Sobol Results with Mathematical Symbols")
    print("="*80)
    
    # Load saved results
    results_df = load_results()
    if results_df is None:
        exit(1)
    
    # Recompute Sobol indices
    print("\nComputing Sobol indices...")
    sobol_results = compute_sobol_indices(results_df)
    
    # Plot with symbols
    print("\nGenerating plots...")
    plot_sobol_results(sobol_results, OUTPUT_DIR)
    plot_parameter_effects(results_df, OUTPUT_DIR)
    plot_heatmap_table(sobol_results, OUTPUT_DIR)
    
    # Create LaTeX table
    create_latex_table(sobol_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Done! All plots saved with mathematical symbols.")
    print("="*80)
