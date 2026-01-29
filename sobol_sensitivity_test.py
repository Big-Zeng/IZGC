"""
Sobol敏感性分析测试脚本
对ContourBasedClusteringWithoutZ22算法的参数进行Sobol敏感性分析

参数范围:
- grid_size: 网格大小，影响虚拟点密度
- max_buffer_distance: 最大缓冲距离
- min_cluster_size: 最小簇大小阈值
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import warnings
import time
import os

import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics

warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
DATASET = "data/2d_3c.csv"                # 数据集路径
NORMALIZE_COORDS = True                   # 是否归一化坐标到[0,100]范围
N_SAMPLES = 64                            # Sobol采样数量 (实际样本数 = N * (2D + 2))
OUTPUT_DIR = "sobol_results"              # 输出目录
# ==================================================

# 定义参数问题
problem = {
    'num_vars': 3,
    'names': ['grid_size', 'radius', 'min_cluster_size'],
    'bounds': [
        [0.5, 5.0],      # grid_size: 0.5 到 5.0
        [5, 50],         # radius: 密度计算半径 5 到 50
        [2, 20]          # min_cluster_size: 2 到 20
    ]
}


def normalize_coordinates(X):
    """归一化坐标到[0, 100]范围"""
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
    运行等值线聚类算法
    
    返回:
        dict: 包含各种评价指标的字典
    """
    try:
        # 计算密度
        points = np.array(X)
        tree = cKDTree(points)
        density = np.array([len(tree.query_ball_point(p, radius)) for p in points])
        
        # 准备数据
        combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
        
        # 创建聚类器 (max_buffer_distance固定为0)
        cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
            combined_data,
            grid_size=grid_size,
            max_buffer_distance=0,
            min_cluster_size=int(min_cluster_size),
            interactive=False,  # 关闭交互模式
            merge_small_clusters=True
        )
        
        # 执行聚类
        cluster.perform_clustering()
        
        # 获取预测标签
        pre_labels = cluster.get_predicted_labels()
        pre_labels = np.array(pre_labels)
        
        # 计算评价指标
        n_clusters = len(set(pre_labels)) - (1 if -1 in pre_labels else 0)
        n_noise = np.sum(pre_labels == -1)
        
        # 如果簇数太少，返回默认值
        if n_clusters < 2:
            return {
                'silhouette': -1,
                'ari': 0,
                'ch_score': 0,
                'db_score': 10,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'success': False
            }
        
        # 计算各种指标
        # 对于有噪声点的情况，只用非噪声点计算轮廓系数
        mask = pre_labels != -1
        if np.sum(mask) > 1 and len(set(pre_labels[mask])) > 1:
            silhouette = silhouette_score(X[mask], pre_labels[mask])
            ch_score = calinski_harabasz_score(X[mask], pre_labels[mask])
            db_score = davies_bouldin_score(X[mask], pre_labels[mask])
        else:
            silhouette = -1
            ch_score = 0
            db_score = 10
        
        ari = adjusted_rand_score(y_true, pre_labels)
        
        return {
            'silhouette': silhouette,
            'ari': ari,
            'ch_score': ch_score,
            'db_score': db_score,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'success': True
        }
        
    except Exception as e:
        print(f"  聚类失败: {e}")
        return {
            'silhouette': -1,
            'ari': 0,
            'ch_score': 0,
            'db_score': 10,
            'n_clusters': 0,
            'n_noise': 0,
            'success': False
        }


def run_sobol_analysis(X, y_true, n_samples=64):
    """
    运行Sobol敏感性分析
    """
    print("\n" + "="*80)
    print("Sobol敏感性分析")
    print("="*80)
    
    # 生成Sobol采样
    print(f"\n生成Sobol采样 (N={n_samples})...")
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
    print(f"总样本数: {len(param_values)}")
    
    # 存储结果
    Y_silhouette = np.zeros(len(param_values))
    Y_ari = np.zeros(len(param_values))
    Y_ch = np.zeros(len(param_values))
    Y_db = np.zeros(len(param_values))
    
    results_list = []
    
    # 运行所有参数组合
    print("\n开始运行聚类测试...")
    start_time = time.time()
    
    for i, params in enumerate(param_values):
        grid_size, radius, min_cluster_size = params
        
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(param_values) - i - 1)
            print(f"  进度: {i+1}/{len(param_values)} ({100*(i+1)/len(param_values):.1f}%) "
                  f"- 已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
        
        result = run_contour_clustering(
            X, y_true,
            grid_size=grid_size,
            radius=radius,
            min_cluster_size=min_cluster_size
        )
        
        Y_silhouette[i] = result['silhouette']
        Y_ari[i] = result['ari']
        Y_ch[i] = result['ch_score']
        Y_db[i] = result['db_score']
        
        results_list.append({
            'grid_size': grid_size,
            'radius': radius,
            'min_cluster_size': min_cluster_size,
            **result
        })
    
    total_time = time.time() - start_time
    print(f"\n完成! 总用时: {total_time:.1f}s")
    
    # 保存详细结果
    results_df = pd.DataFrame(results_list)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(f"{OUTPUT_DIR}/sobol_all_results.csv", index=False)
    print(f"\n详细结果已保存到: {OUTPUT_DIR}/sobol_all_results.csv")
    
    # Sobol分析
    print("\n" + "-"*60)
    print("Sobol敏感性指数分析")
    print("-"*60)
    
    metrics = {
        'Silhouette': Y_silhouette,
        'ARI': Y_ari,
        'CH_Score': Y_ch,
        'DB_Score': Y_db
    }
    
    sobol_results = {}
    
    for metric_name, Y in metrics.items():
        # 处理无效值
        Y_clean = np.nan_to_num(Y, nan=0, posinf=0, neginf=0)
        
        # 检查是否有足够的变化
        if np.std(Y_clean) < 1e-10:
            print(f"\n{metric_name}: 数据变化太小，跳过分析")
            continue
        
        try:
            Si = sobol.analyze(problem, Y_clean, calc_second_order=True)
            sobol_results[metric_name] = Si
            
            print(f"\n{metric_name}:")
            print(f"  一阶敏感性指数 (S1):")
            for j, name in enumerate(problem['names']):
                print(f"    {name}: {Si['S1'][j]:.4f} (置信区间: {Si['S1_conf'][j]:.4f})")
            
            print(f"  总敏感性指数 (ST):")
            for j, name in enumerate(problem['names']):
                print(f"    {name}: {Si['ST'][j]:.4f} (置信区间: {Si['ST_conf'][j]:.4f})")
                
        except Exception as e:
            print(f"\n{metric_name}: 分析失败 - {e}")
    
    return sobol_results, results_df


def plot_sobol_results(sobol_results, output_dir):
    """绘制Sobol敏感性分析结果"""
    if not sobol_results:
        print("没有有效的Sobol结果可绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    param_names = problem['names']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (metric_name, Si) in enumerate(sobol_results.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        # 绘制S1和ST
        bars1 = ax.bar(x - width/2, Si['S1'], width, label='S1 (一阶)', 
                       color=colors, alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, Si['ST'], width, label='ST (总效应)', 
                       color=colors, alpha=0.4, edgecolor='black', hatch='//')
        
        # 添加误差线
        ax.errorbar(x - width/2, Si['S1'], yerr=Si['S1_conf'], fmt='none', 
                   color='black', capsize=3)
        ax.errorbar(x + width/2, Si['ST'], yerr=Si['ST_conf'], fmt='none', 
                   color='black', capsize=3)
        
        ax.set_ylabel('敏感性指数')
        ax.set_title(f'{metric_name} 敏感性分析')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sobol_sensitivity.png", dpi=150, bbox_inches='tight')
    print(f"\n敏感性分析图已保存到: {output_dir}/sobol_sensitivity.png")
    plt.show()


def plot_parameter_effects(results_df, output_dir):
    """绘制参数对指标的影响"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    params = ['grid_size', 'radius', 'min_cluster_size']
    metrics = ['silhouette', 'ari', 'ch_score', 'db_score']
    metric_labels = ['Silhouette', 'ARI', 'CH Score', 'DB Score']
    
    for i, param in enumerate(params):
        for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]
            
            # 过滤有效数据
            valid_mask = results_df['success'] == True
            if metric == 'silhouette':
                valid_mask &= results_df[metric] > -1
            
            x = results_df.loc[valid_mask, param]
            y = results_df.loc[valid_mask, metric]
            
            ax.scatter(x, y, alpha=0.5, s=20)
            
            # 添加趋势线
            if len(x) > 2:
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7)
            
            ax.set_xlabel(param)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    
    plt.suptitle('参数对聚类指标的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_effects.png", dpi=150, bbox_inches='tight')
    print(f"参数影响图已保存到: {output_dir}/parameter_effects.png")
    plt.show()


def find_best_parameters(results_df):
    """找出最佳参数组合"""
    print("\n" + "="*80)
    print("最佳参数组合")
    print("="*80)
    
    # 过滤成功的结果
    valid_df = results_df[results_df['success'] == True].copy()
    
    if len(valid_df) == 0:
        print("没有成功的聚类结果")
        return
    
    # 按不同指标找最佳
    metrics_info = [
        ('silhouette', '最高', True),
        ('ari', '最高', True),
        ('ch_score', '最高', True),
        ('db_score', '最低', False)
    ]
    
    for metric, direction, ascending in metrics_info:
        if metric == 'silhouette':
            valid_metric = valid_df[valid_df[metric] > -1]
        else:
            valid_metric = valid_df
        
        if len(valid_metric) == 0:
            continue
            
        if ascending:
            best_idx = valid_metric[metric].idxmax()
        else:
            best_idx = valid_metric[metric].idxmin()
        
        best = valid_metric.loc[best_idx]
        
        print(f"\n{metric} {direction}:")
        print(f"  grid_size: {best['grid_size']:.3f}")
        print(f"  radius: {best['radius']:.1f}")
        print(f"  min_cluster_size: {best['min_cluster_size']:.0f}")
        print(f"  {metric}: {best[metric]:.4f}")
        print(f"  簇数: {best['n_clusters']:.0f}, 噪声点: {best['n_noise']:.0f}")


# 主程序
if __name__ == "__main__":
    print("="*80)
    print("ContourBasedClustering Sobol敏感性分析")
    print("="*80)
    
    # 加载数据
    print(f"\n加载数据集: {DATASET}")
    df = pd.read_csv(DATASET)
    X = df[['x', 'y']].astype(float).values
    y_true = df['target'].values
    
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    # 归一化
    if NORMALIZE_COORDS:
        print("\n归一化坐标到[0, 100]范围...")
        X = normalize_coordinates(X)
    
    # 运行Sobol分析
    sobol_results, results_df = run_sobol_analysis(X, y_true, n_samples=N_SAMPLES)
    
    # 绘制结果
    plot_sobol_results(sobol_results, OUTPUT_DIR)
    plot_parameter_effects(results_df, OUTPUT_DIR)
    
    # 找出最佳参数
    find_best_parameters(results_df)
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
