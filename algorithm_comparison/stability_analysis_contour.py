"""
ContourBasedClustering 稳定性分析
评估不同 grid_size 参数下的聚类性能稳定性
生成科研级别的 ARI 变化图
"""

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
import sys
import time
from scipy.spatial import cKDTree

# Apply publication-grade plotting settings
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ContourBasedClusteringWithoutZ22 as ContourClustering
from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates, evaluate_clustering
)
from clustering_metrics import ClusteringMetrics

# ==================== Configuration ====================
# Dataset path (ground-truth labels required)
DATASET = "../data/2d_3c.csv"
OUTPUT_DIR = "results_stability"
NORMALIZE = True

# grid_size sweep range (start, end, step)
GRID_SIZE_RANGE = (0.5, 5.0, 0.001)

# Fix the remaining parameters
FIXED_PARAMS = {
    'max_buffer_distance': 0,
    'min_cluster_size': 5,
    'density_radius': 30,
    'merge_small_clusters': True
}

# Evaluation metrics
METRICS_TO_PLOT = ['ARI', 'AMI', 'FMI']  # Options: 'ARI', 'AMI', 'FMI', 'Silhouette' or 'CH'


# ==============================================


def generate_grid_size_values(start, end, step):
    """生成 grid_size 值列表"""
    values = []
    current = start
    while current <= end + step / 2:
        values.append(round(current, 3))
        current += step
    return values


def run_contour_clustering_single(X, grid_size, fixed_params):
    """运行单次等值线聚类"""
    grid_size = round(grid_size, 3)

    # Compute density
    tree = cKDTree(X)
    density = np.array([len(tree.query_ball_point(p, fixed_params['density_radius'])) for p in X])

    # Prepare data as (x, y, density, index)
    combined_data = [(*point, density[i], i) for i, point in enumerate(X)]

    # Create clustering instance
    cluster = ContourClustering.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=grid_size,
        max_buffer_distance=fixed_params['max_buffer_distance'],
        min_cluster_size=fixed_params['min_cluster_size'],
        merge_small_clusters=fixed_params['merge_small_clusters'],
        interactive=False  # Disable interactive mode
    )

    # Run clustering
    cluster.perform_clustering()

    # Get labels
    labels = cluster.get_predicted_labels()

    return np.array(labels)


def evaluate_stability(X, y_true, grid_size_values, fixed_params):
    """评估不同 grid_size 下的聚类稳定性"""
    results = []

    print(f"\n开始稳定性评估...")
    print(f"grid_size 范围: {grid_size_values[0]:.3f} -> {grid_size_values[-1]:.3f}")
    print(f"总评估次数: {len(grid_size_values)}")
    print(f"固定参数: {fixed_params}")

    start_time = time.time()

    for i, grid_size in enumerate(grid_size_values, 1):
        try:
            labels = run_contour_clustering_single(X, grid_size, fixed_params)

            # Check whether the clustering is valid
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(np.array(labels) == -1))
            # Compute evaluation metrics
            metrics = evaluate_clustering(y_true, labels, X)
            result = {
                'grid_size': grid_size,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'ARI': metrics['ARI'],
                'AMI': metrics['AMI'],
                'FMI': metrics['FMI'],
                'Silhouette': metrics.get('Silhouette', metrics.get('Silhouette_core', np.nan)),
                'CH': metrics['CH'],
                'valid': True
            }
            # if n_clusters < 2:
            # Invalid clustering: record NaN
            #     result = {
            #         'grid_size': grid_size,
            #         'n_clusters': n_clusters,
            #         'n_noise': n_noise,
            #         'ARI': np.nan,
            #         'AMI': np.nan,
            #         'FMI': np.nan,
            #         'Silhouette': np.nan,
            #         'CH': np.nan,
            #         'valid': False
            #     }
            # else:
            # Compute evaluation metrics
            #     metrics = evaluate_clustering(y_true, labels, X)
            #     result = {
            #         'grid_size': grid_size,
            #         'n_clusters': n_clusters,
            #         'n_noise': n_noise,
            #         'ARI': metrics['ARI'],
            #         'AMI': metrics['AMI'],
            #         'FMI': metrics['FMI'],
            #         'Silhouette': metrics.get('Silhouette', metrics.get('Silhouette_core', np.nan)),
            #         'CH': metrics['CH'],
            #         'valid': True
            #     }

            results.append(result)

            if i % 10 == 0 or i == len(grid_size_values):
                elapsed = time.time() - start_time
                valid_count = sum(1 for r in results if r['valid'])
                print(f"  进度: {i}/{len(grid_size_values)}, 有效: {valid_count}, 用时: {elapsed:.1f}s")

        except Exception as e:
            # Record the failure
            result = {
                'grid_size': grid_size,
                'n_clusters': 0,
                'n_noise': 0,
                'ARI': np.nan,
                'AMI': np.nan,
                'FMI': np.nan,
                'Silhouette': np.nan,
                'CH': np.nan,
                'valid': False,
                'error': str(e)[:50]
            }
            results.append(result)

    elapsed = time.time() - start_time
    valid_count = sum(1 for r in results if r['valid'])
    print(f"\n评估完成: 有效 {valid_count}/{len(grid_size_values)}, 总用时: {elapsed:.2f}s")

    return results


def plot_stability_curve(results, dataset_name, output_dir, metrics_to_plot):
    """绘制稳定性曲线（科研级别）"""
    df = pd.DataFrame(results)

    # Keep the valid results
    df_valid = df[df['valid'] == True].copy()

    if len(df_valid) == 0:
        print("警告: 没有有效结果，无法绘图")
        return

    # 1. Plot the multi-metric comparison (untitled)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, metric in enumerate(metrics_to_plot):
        if metric in df_valid.columns:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            ax.plot(df_valid['grid_size'], df_valid[metric],
                    color=color, marker=marker, markersize=4,
                    linewidth=1.5, label=metric, alpha=0.8)

    ax.set_xlabel('Grid Size', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    # Do not set a title
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(df_valid['grid_size'].min() - 0.1, df_valid['grid_size'].max() + 0.1)
    ax.set_ylim(-0.1, 1.1)

    # Annotate the best point
    best_idx = df_valid['ARI'].idxmax()
    best_grid_size = df_valid.loc[best_idx, 'grid_size']
    best_ari = df_valid.loc[best_idx, 'ARI']
    ax.axvline(x=best_grid_size, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate(f'Best: {best_grid_size:.3f}\nARI={best_ari:.4f}',
                xy=(best_grid_size, best_ari),
                xytext=(best_grid_size + 0.3, best_ari - 0.15),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    plt.tight_layout()
    save_path = f"{output_dir}/stability_metrics_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {save_path}")

    # 2. Plot the ARI alone (untitled)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_valid['grid_size'], df_valid['ARI'],
            color='#1f77b4', marker='o', markersize=5,
            linewidth=2, label='ARI', alpha=0.9)

    # Fill the confidence band from a rolling-window standard deviation
    window_size = 5
    if len(df_valid) > window_size:
        rolling_std = df_valid['ARI'].rolling(window=window_size, center=True).std().fillna(0)
        ax.fill_between(df_valid['grid_size'],
                        df_valid['ARI'] - rolling_std,
                        df_valid['ARI'] + rolling_std,
                        color='#1f77b4', alpha=0.2, label='Std Dev')

    ax.set_xlabel('Grid Size', fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontweight='bold')
    # Do not set a title
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(df_valid['grid_size'].min() - 0.1, df_valid['grid_size'].max() + 0.1)

    # Annotate the best point
    ax.scatter([best_grid_size], [best_ari], color='red', s=100, zorder=5, marker='*', label='Best')
    ax.axvline(x=best_grid_size, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=best_ari, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    # Add text annotations
    ax.text(0.02, 0.98, f'Best grid_size: {best_grid_size:.3f}\nBest ARI: {best_ari:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_path = f"{output_dir}/stability_ARI_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {save_path}")

    # 3. Plot the number of clusters (untitled, without the noise curve)
    fig, ax = plt.subplots(figsize=(10, 6))

    color1 = '#1f77b4'
    ax.set_xlabel('Grid Size', fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontweight='bold')
    ax.plot(df_valid['grid_size'], df_valid['n_clusters'],
            color=color1, marker='o', markersize=4, linewidth=1.5, label='Clusters')
    ax.set_xlim(df_valid['grid_size'].min() - 0.1, df_valid['grid_size'].max() + 0.1)

    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    save_path = f"{output_dir}/stability_clusters_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {save_path}")

    return df_valid


def save_results_to_csv(results, dataset_name, output_dir):
    """保存结果到CSV文件"""
    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/stability_results_{dataset_name}.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  已保存: {csv_path}")
    return df


def print_summary(df, dataset_name):
    """打印统计摘要"""
    df_valid = df[df['valid'] == True]

    print(f"\n{'=' * 70}")
    print(f"稳定性分析摘要 - {dataset_name}")
    print(f"{'=' * 70}")

    print(f"\n有效评估次数: {len(df_valid)}/{len(df)}")

    if len(df_valid) > 0:
        print(f"\nARI 统计:")
        print(
            f"  最大值: {df_valid['ARI'].max():.4f} (grid_size={df_valid.loc[df_valid['ARI'].idxmax(), 'grid_size']:.3f})")
        print(
            f"  最小值: {df_valid['ARI'].min():.4f} (grid_size={df_valid.loc[df_valid['ARI'].idxmin(), 'grid_size']:.3f})")
        print(f"  平均值: {df_valid['ARI'].mean():.4f}")
        print(f"  标准差: {df_valid['ARI'].std():.4f}")
        print(f"  中位数: {df_valid['ARI'].median():.4f}")

        # Identify the stable range where ARI > 0.9
        high_ari = df_valid[df_valid['ARI'] > 0.9]
        if len(high_ari) > 0:
            print(f"\nARI > 0.9 的 grid_size 范围:")
            print(f"  {high_ari['grid_size'].min():.3f} ~ {high_ari['grid_size'].max():.3f} ({len(high_ari)} 个值)")

        # Identify the stable range where ARI > 0.8
        good_ari = df_valid[df_valid['ARI'] > 0.8]
        if len(good_ari) > 0:
            print(f"\nARI > 0.8 的 grid_size 范围:")
            print(f"  {good_ari['grid_size'].min():.3f} ~ {good_ari['grid_size'].max():.3f} ({len(good_ari)} 个值)")

        print(f"\n簇数量统计:")
        print(f"  范围: {df_valid['n_clusters'].min()} ~ {df_valid['n_clusters'].max()}")
        print(f"  平均: {df_valid['n_clusters'].mean():.1f}")

    print(f"{'=' * 70}")


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ContourBasedClustering 稳定性分析")
    print("=" * 70)

    # Load data
    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

    print(f"\n数据集: {dataset_name}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")

    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")

    # Generate the grid_size values
    grid_size_values = generate_grid_size_values(*GRID_SIZE_RANGE)
    print(f"\ngrid_size 扫描: {GRID_SIZE_RANGE[0]:.3f} -> {GRID_SIZE_RANGE[1]:.3f}, 步长={GRID_SIZE_RANGE[2]:.3f}")
    print(f"总共 {len(grid_size_values)} 个值")

    # Assess stability
    results = evaluate_stability(X, y_true, grid_size_values, FIXED_PARAMS)

    # Save results to CSV
    df = save_results_to_csv(results, dataset_name, OUTPUT_DIR)

    # Plot the stability curve
    print("\n生成图表...")
    df_valid = plot_stability_curve(results, dataset_name, OUTPUT_DIR, METRICS_TO_PLOT)

    # Print the summary
    print_summary(df, dataset_name)

    # Save the summary to txt
    summary_path = f"{OUTPUT_DIR}/stability_summary_{dataset_name}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        df_valid = df[df['valid'] == True]
        f.write(f"ContourBasedClustering 稳定性分析摘要\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"样本数: {len(X)}\n")
        f.write(f"真实簇数: {len(np.unique(y_true))}\n\n")
        f.write(
            f"grid_size 扫描范围: {GRID_SIZE_RANGE[0]:.3f} -> {GRID_SIZE_RANGE[1]:.3f}, 步长={GRID_SIZE_RANGE[2]:.3f}\n")
        f.write(f"固定参数: {FIXED_PARAMS}\n\n")
        f.write(f"有效评估次数: {len(df_valid)}/{len(df)}\n\n")

        if len(df_valid) > 0:
            f.write(f"ARI 统计:\n")
            f.write(
                f"  最大值: {df_valid['ARI'].max():.4f} (grid_size={df_valid.loc[df_valid['ARI'].idxmax(), 'grid_size']:.3f})\n")
            f.write(
                f"  最小值: {df_valid['ARI'].min():.4f} (grid_size={df_valid.loc[df_valid['ARI'].idxmin(), 'grid_size']:.3f})\n")
            f.write(f"  平均值: {df_valid['ARI'].mean():.4f}\n")
            f.write(f"  标准差: {df_valid['ARI'].std():.4f}\n")
            f.write(f"  中位数: {df_valid['ARI'].median():.4f}\n\n")

            high_ari = df_valid[df_valid['ARI'] > 0.9]
            if len(high_ari) > 0:
                f.write(
                    f"ARI > 0.9 的 grid_size 范围: {high_ari['grid_size'].min():.3f} ~ {high_ari['grid_size'].max():.3f} ({len(high_ari)} 个值)\n")

            good_ari = df_valid[df_valid['ARI'] > 0.8]
            if len(good_ari) > 0:
                f.write(
                    f"ARI > 0.8 的 grid_size 范围: {good_ari['grid_size'].min():.3f} ~ {good_ari['grid_size'].max():.3f} ({len(good_ari)} 个值)\n")

    print(f"  已保存: {summary_path}")

    print(f"\n完成！所有结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
