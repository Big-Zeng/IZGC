"""
通用工具函数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering_metrics import ClusteringMetrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, davies_bouldin_score


def load_dataset(filepath):
    """加载数据集"""
    df = pd.read_csv(filepath)
    X = df[['x', 'y']].values.astype(float)
    # 检查是否有 target 列（真实标签）
    if 'target' in df.columns:
        y_true = df['target'].values
    else:
        y_true = None  # 无监督数据集
    return X, y_true


def load_real_dataset(filepath):
    """加载真实数据集（无标签）"""
    df = pd.read_csv(filepath)
    X = df[['x', 'y']].values.astype(float)
    return X


def normalize_coordinates(X):
    """归一化坐标到[0, 100]范围"""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    X_norm = np.zeros_like(X)
    X_norm[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 50
    X_norm[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 50
    
    return X_norm


def calculate_inertia(X, labels):
    """计算簇内误差平方和"""
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


def evaluate_clustering(y_true, y_pred, X):
    """
    评估聚类结果（有监督指标 + 无监督指标）
    
    使用加权复合评估方法（论文严谨做法）：
    - Silhouette_core: 仅对非噪声点计算轮廓系数（聚类质量）
    - Outlier_ratio: 噪声点占比（覆盖率）
    - Composite: 加权复合得分 = Silhouette_core * (1 - Outlier_ratio)
    """
    y_pred_array = np.array(y_pred)
    n_total = len(y_pred_array)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = int(np.sum(y_pred_array == -1))
    
    # 核心点（非噪声点）
    mask_core = y_pred_array != -1
    n_core = int(np.sum(mask_core))
    outlier_ratio = n_noise / n_total if n_total > 0 else 1.0
    
    # 轮廓系数（仅对核心点计算）
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
    
    # 复合得分（论文严谨做法）
    # Composite = Silhouette_core * (1 - Outlier_ratio)
    # 这样既考虑聚类质量，又惩罚过多的噪声点
    if silhouette_core > -1:
        composite_score = silhouette_core * (1 - outlier_ratio)
    else:
        composite_score = -1
    
    result = {
        'Silhouette_core': silhouette_core,  # 核心点轮廓系数
        'Silhouette': silhouette_core,       # 兼容旧代码（别名）
        'Outlier_ratio': outlier_ratio,      # 噪声点占比
        'Composite': composite_score,         # 复合得分（主要优化目标）
        'CH': ch_score,                       # Calinski-Harabasz 指数
        'DBI': dbi_score,                     # Davies-Bouldin 指数（越小越好）
        'Inertia': calculate_inertia(X, y_pred_array),
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    # 有监督指标（仅当有真实标签时计算）
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
    """仅使用无监督指标评估聚类结果"""
    return evaluate_clustering(None, y_pred, X)


def plot_clustering_result(X, labels, title, save_path, figsize=(8, 6), dpi=300, 
                           marker_size=30, show_legend=False, show_title=False):
    """
    绘制聚类结果图（论文级别）
    
    参数:
        X: 坐标数组
        labels: 聚类标签
        title: 图表标题（仅当show_title=True时显示）
        save_path: 保存路径
        figsize: 图表大小
        dpi: 分辨率（论文通常需要300dpi）
        marker_size: 点大小
        show_legend: 是否显示图例（默认False）
        show_title: 是否显示标题（默认False）
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # 固定颜色列表（20种不同颜色，适合论文）
    FIXED_COLORS = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿色
        '#17becf',  # 青色
        '#aec7e8',  # 浅蓝
        '#ffbb78',  # 浅橙
        '#98df8a',  # 浅绿
        '#ff9896',  # 浅红
        '#c5b0d5',  # 浅紫
        '#c49c94',  # 浅棕
        '#f7b6d2',  # 浅粉
        '#c7c7c7',  # 浅灰
        '#dbdb8d',  # 浅黄绿
        '#9edae5',  # 浅青
    ]
    
    # 获取唯一标签（排序以保证一致性）
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # 绘制每个簇
    for label in unique_labels:
        mask = np.array(labels) == label
        if label == -1:
            # 噪声点用灰色X标记
            plt.scatter(X[mask, 0], X[mask, 1], c='gray', s=marker_size/2, 
                       alpha=0.5, label='Noise', marker='x')
        else:
            # 使用固定颜色（循环使用）
            color = FIXED_COLORS[label % len(FIXED_COLORS)]
            plt.scatter(X[mask, 0], X[mask, 1], c=color, 
                       s=marker_size, alpha=0.7, label=f'Cluster {label+1}',
                       edgecolors='white', linewidths=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if show_title:
        plt.title(title)
    
    if show_legend and n_clusters:
       plt.legend(loc='best', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  图表已保存: {save_path}")

def plot_ground_truth(X, y_true, save_path, figsize=(8, 6), dpi=300, marker_size=30):
    """绘制真实标签图"""
    plot_clustering_result(X, y_true, 'Ground Truth', save_path, 
                          figsize=figsize, dpi=dpi, marker_size=marker_size)


def print_best_result(algorithm_name, best_params, best_metrics):
    """打印最佳结果"""
    print(f"\n{'='*60}")
    print(f"{algorithm_name} 最佳结果")
    print(f"{'='*60}")
    print(f"最佳参数: {best_params}")
    
    # 有监督指标（如果有）
    if best_metrics.get('ARI') is not None:
        print(f"AMI: {best_metrics['AMI']:.4f}")
        print(f"ARI: {best_metrics['ARI']:.4f}")
        print(f"FMI: {best_metrics['FMI']:.4f}")
    
    # 无监督指标（复合评估）
    if 'Composite' in best_metrics:
        print(f"Composite: {best_metrics['Composite']:.4f} (主要指标)")
        print(f"  ├─ Silhouette_core: {best_metrics['Silhouette_core']:.4f}")
        print(f"  └─ Outlier_ratio: {best_metrics['Outlier_ratio']:.4f}")
    elif 'Silhouette' in best_metrics:
        print(f"Silhouette: {best_metrics['Silhouette']:.4f}")
    
    print(f"CH: {best_metrics['CH']:.2f}")
    if 'DBI' in best_metrics:
        dbi_val = best_metrics['DBI']
        if dbi_val == float('inf'):
            print(f"DBI: inf")
        else:
            print(f"DBI: {dbi_val:.4f} (越小越好)")
    print(f"簇数: {best_metrics['n_clusters']}, 噪声点: {best_metrics['n_noise']}")
    print(f"{'='*60}")


def save_best_result(algorithm_name, dataset_name, best_params, best_metrics, save_path):
    """
    保存最佳结果到txt文件
    
    参数:
        algorithm_name: 算法名称
        dataset_name: 数据集名称
        best_params: 最佳参数字典
        best_metrics: 最佳指标字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"算法: {algorithm_name}\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("最佳参数:\n")
        for key, value in best_params.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\n评估指标:\n")
        
        # 有监督指标（如果有）
        if best_metrics.get('ARI') is not None:
            f.write(f"  AMI: {best_metrics['AMI']:.4f}\n")
            f.write(f"  ARI: {best_metrics['ARI']:.4f}\n")
            f.write(f"  FMI: {best_metrics['FMI']:.4f}\n")
        
        # 无监督指标
        if 'Composite' in best_metrics:
            f.write(f"  Composite: {best_metrics['Composite']:.4f} (主要指标)\n")
            f.write(f"    ├─ Silhouette_core: {best_metrics['Silhouette_core']:.4f}\n")
            f.write(f"    └─ Outlier_ratio: {best_metrics['Outlier_ratio']:.4f}\n")
        elif 'Silhouette' in best_metrics:
            f.write(f"  Silhouette: {best_metrics['Silhouette']:.4f}\n")
        
        f.write(f"  CH: {best_metrics['CH']:.2f}\n")
        if 'DBI' in best_metrics:
            dbi_val = best_metrics['DBI']
            if dbi_val == float('inf'):
                f.write(f"  DBI: inf\n")
            else:
                f.write(f"  DBI: {dbi_val:.4f} (越小越好)\n")
        f.write(f"  簇数: {best_metrics['n_clusters']}\n")
        f.write(f"  噪声点: {best_metrics['n_noise']}\n")
        f.write(f"{'='*60}\n")
    
    print(f"  结果已保存: {save_path}")
