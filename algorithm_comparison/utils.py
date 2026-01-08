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
from sklearn.metrics import silhouette_score


def load_dataset(filepath):
    """加载数据集"""
    df = pd.read_csv(filepath)
    X = df[['x', 'y']].values.astype(float)
    y_true = df['target'].values
    return X, y_true


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
    """评估聚类结果"""
    metrics = ClusteringMetrics.evaluate_all(y_true, y_pred)
    
    # 轮廓系数
    if len(set(y_pred)) > 1:
        mask = np.array(y_pred) != -1
        if np.sum(mask) > 0 and len(np.unique(np.array(y_pred)[mask])) > 1:
            silhouette = silhouette_score(X[mask], np.array(y_pred)[mask])
        else:
            silhouette = -1
    else:
        silhouette = -1
    
    return {
        'AMI': metrics['AMI'],
        'ARI': metrics['ARI'],
        'FMI': metrics['FMI'],
        'Silhouette': silhouette,
        'Inertia': calculate_inertia(X, np.array(y_pred)),
        'n_clusters': len(set(y_pred)) - (1 if -1 in y_pred else 0),
        'n_noise': list(y_pred).count(-1)
    }


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
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters + 1)))
    
    # 绘制每个簇
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # 噪声点用灰色
            plt.scatter(X[mask, 0], X[mask, 1], c='gray', s=marker_size/2, 
                       alpha=0.5, label='Noise', marker='x')
        else:
            plt.scatter(X[mask, 0], X[mask, 1], c=[colors[label % len(colors)]], 
                       s=marker_size, alpha=0.7, label=f'Cluster {label+1}',
                       edgecolors='white', linewidths=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if show_title:
        plt.title(title)
    
    if show_legend and n_clusters <= 10:
        plt.legend(loc='best', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
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
    print(f"AMI: {best_metrics['AMI']:.4f}")
    print(f"ARI: {best_metrics['ARI']:.4f}")
    print(f"FMI: {best_metrics['FMI']:.4f}")
    print(f"Silhouette: {best_metrics['Silhouette']:.4f}")
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
        f.write(f"  AMI: {best_metrics['AMI']:.4f}\n")
        f.write(f"  ARI: {best_metrics['ARI']:.4f}\n")
        f.write(f"  FMI: {best_metrics['FMI']:.4f}\n")
        f.write(f"  Silhouette: {best_metrics['Silhouette']:.4f}\n")
        f.write(f"  簇数: {best_metrics['n_clusters']}\n")
        f.write(f"  噪声点: {best_metrics['n_noise']}\n")
        f.write(f"{'='*60}\n")
    
    print(f"  结果已保存: {save_path}")
