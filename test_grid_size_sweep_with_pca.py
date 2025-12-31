"""
Grid Size扫描测试（支持高维数据PCA降维）
从0.01开始递增grid_size，找到ARI最大的值
支持CSV、TXT和MAT文件，自动归一化坐标
对于高维数据（>2维），自动使用PCA降维到2维
"""

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import os
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics


# ==================== 配置区域 ====================
DATASET = "compare/dataset/uci/iris.txt"    # 数据集路径 (支持.csv, .txt或.mat)
# GRID_SIZE_START 和 GRID_SIZE_END 将自动根据数据范围计算
# GRID_SIZE_START = 最短边 / 100
# GRID_SIZE_END = 最短边 / 2
GRID_SIZE_STEP = 0.001                  # 步长
MAX_BUFFER_DISTANCE = 0                 # 最大缓冲距离
INTERPOLATION_METHOD = 'linear'         # 插值方法: linear, cubic, idw

# 归一化选项
NORMALIZE_COORDS = True                 # 是否归一化坐标到[0,100]范围

# PCA选项
USE_PCA = True                          # 是否对高维数据使用PCA降维
PCA_N_COMPONENTS = 2                    # PCA降维目标维度

# 是否显示每次测试的可视化（设为False可加快速度）
SHOW_VISUALIZATION = False
# ==================================================


def load_dataset(filepath):
    """
    加载数据集（支持CSV、TXT和MAT文件）
    
    参数:
        filepath: 文件路径
        
    返回:
        X: 特征数组 (n_samples, n_features)
        y_true: 真实标签
    """
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        # 读取CSV文件
        df = pd.read_csv(filepath)
        # 假设最后一列是标签
        X = df.iloc[:, :-1].values
        y_true = df.iloc[:, -1].values
        
    elif file_ext == '.txt':
        # 读取TXT文件（逗号或制表符分隔）
        try:
            # 尝试逗号分隔
            data = np.loadtxt(filepath, delimiter=',')
        except:
            # 尝试制表符分隔
            data = np.loadtxt(filepath, delimiter='\t')
        
        # 假设最后一列是标签
        X = data[:, :-1]
        y_true = data[:, -1]
        
    elif file_ext == '.mat':
        # 读取MAT文件
        mat_data = loadmat(filepath)
        
        # 尝试常见的变量名
        data_array = None
        if 'd' in mat_data:
            data_array = mat_data['d']
        elif 'X' in mat_data:
            data_array = mat_data['X']
        elif 'data' in mat_data:
            data_array = mat_data['data']
        elif 'points' in mat_data:
            data_array = mat_data['points']
        else:
            # 找到第一个非系统变量
            for key in mat_data.keys():
                if not key.startswith('__'):
                    data_array = mat_data[key]
                    break
        
        if data_array is None:
            raise ValueError("MAT文件中找不到数据")
        
        # 确保是2D数组
        if data_array.ndim == 1:
            raise ValueError("MAT文件数据维度不正确")
        
        # 提取特征和标签
        if 'y' in mat_data:
            X = data_array
            y_true = mat_data['y'].flatten()
        elif 'labels' in mat_data:
            X = data_array
            y_true = mat_data['labels'].flatten()
        elif 'target' in mat_data:
            X = data_array
            y_true = mat_data['target'].flatten()
        else:
            # 假设最后一列是标签
            X = data_array[:, :-1]
            y_true = data_array[:, -1].flatten()
        
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持.csv, .txt和.mat")
    
    # 确保数据类型正确
    X = np.array(X, dtype=float)
    y_true = np.array(y_true, dtype=int)
    
    return X, y_true


def apply_pca(X, n_components=2):
    """
    使用PCA降维
    
    参数:
        X: 特征数组 (n_samples, n_features)
        n_components: 目标维度
        
    返回:
        X_pca: 降维后的数据 (n_samples, n_components)
        pca: PCA对象
        explained_variance_ratio: 解释方差比例
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca, pca.explained_variance_ratio_


def normalize_coordinates(X):
    """
    归一化坐标到[0, 100]范围
    
    参数:
        X: 坐标数组 (n_samples, 2)
        
    返回:
        X_normalized: 归一化后的坐标
        scale_info: 缩放信息字典
    """
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 归一化到[0, 100]
    X_normalized = np.zeros_like(X)
    X_normalized[:, 0] = ((X[:, 0] - x_min) / x_range * 100) if x_range > 0 else 0
    X_normalized[:, 1] = ((X[:, 1] - y_min) / y_range * 100) if y_range > 0 else 0
    
    scale_info = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'x_range': x_range,
        'y_range': y_range
    }
    
    return X_normalized, scale_info


def calculate_grid_size_range(X):
    """
    根据数据边界框自动计算grid_size范围
    
    参数:
        X: 坐标数组 (n_samples, 2)

    返回:
        grid_size_start: 起始grid_size (最短边 / 100)
        grid_size_end: 结束grid_size (最短边 / 2)
        min_edge: 最短边长度
    """
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 找到最短边
    min_edge = min(x_range, y_range)
    
    # 计算grid_size范围
    grid_size_start = 2.5
    grid_size_end = 4
    
    return grid_size_start, grid_size_end, min_edge


class GridSizeTester:
    """Grid Size测试器"""
    
    def __init__(self, coords, true_labels, normalized=False, scale_info=None, 
                 pca_info=None):
        self.coords = coords
        self.true_labels = true_labels
        self.normalized = normalized
        self.scale_info = scale_info
        self.pca_info = pca_info
        self.results = []
    
    def calculate_inertia(self, points, labels):
        """计算簇内误差平方和"""
        inertia = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            if len(cluster_points) == 0:
                continue
            cluster_center = np.mean(cluster_points, axis=0)
            inertia += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
        return inertia
    
    def test_grid_size(self, grid_size, max_buffer_distance, interpolation_method='linear'):
        """测试单个grid_size值"""
        # 保留3位小数
        grid_size = round(grid_size, 3)
        
        try:
            # 计算密度（使用Voronoi）
            points = self.coords
            vor = Voronoi(points)
            tree = cKDTree(points)
            density = np.array([len(tree.query_ball_point(p, 30)) for p in points])
            
            # 准备数据
            combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
            
            # 禁用可视化（加快速度）
            import matplotlib
            if not SHOW_VISUALIZATION:
                matplotlib.use('Agg')
            
            # 创建聚类实例
            cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
                combined_data,
                grid_size=grid_size,
                max_buffer_distance=max_buffer_distance,
                min_cluster_size=0
            )
            
            cluster.perform_clustering()
            
            # 获取标签
            pred_labels = cluster.get_predicted_labels()
            
            # 统计信息
            n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
            n_noise = list(pred_labels).count(-1)
            
            # 检查聚类是否有效
            if n_clusters == 0:
                return {
                    'success': False,
                    'grid_size': grid_size,
                    'error': f'所有点都被标记为噪声 (n_noise={n_noise})'
                }
            
            if n_clusters == 1 and n_noise == 0:
                return {
                    'success': False,
                    'grid_size': grid_size,
                    'error': f'只产生了一个簇，没有分离 (n_points={len(pred_labels)})'
                }
            
            # 计算指标
            metrics = ClusteringMetrics.evaluate_all(self.true_labels, pred_labels)
            inertia = self.calculate_inertia(self.coords, pred_labels)
            
            # 计算轮廓系数
            from sklearn.metrics import silhouette_score
            if len(set(pred_labels)) > 1:
                mask = np.array(pred_labels) != -1
                if np.sum(mask) > 0 and len(np.unique(np.array(pred_labels)[mask])) > 1:
                    silhouette = silhouette_score(self.coords[mask], np.array(pred_labels)[mask])
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            return {
                'success': True,
                'grid_size': grid_size,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'AMI': metrics['AMI'],
                'ARI': metrics['ARI'],
                'FMI': metrics['FMI'],
                'Silhouette': silhouette,
                'Inertia': inertia,
                'labels': pred_labels
            }
            
        except Exception as e:
            return {
                'success': False,
                'grid_size': grid_size,
                'error': str(e)
            }
    
    def sweep_grid_size(self, start, end, step, max_buffer_distance, interpolation_method='linear'):
        """扫描grid_size范围"""
        print(f"开始扫描 grid_size 从 {start:.3f} 到 {end:.3f}，步长 {step}")
        print(f"插值方法: {interpolation_method}")
        print(f"最大缓冲距离: {max_buffer_distance}")
        print("="*80)
        
        grid_sizes = np.arange(start, end + step, step)
        total = len(grid_sizes)
        
        success_count = 0
        fail_count = 0
        
        for i, grid_size in enumerate(grid_sizes, 1):
            print(f"\n[{i}/{total}] 测试 grid_size = {grid_size:.3f}...", end=' ')
            
            start_time = time.time()
            result = self.test_grid_size(grid_size, max_buffer_distance, interpolation_method)
            elapsed = time.time() - start_time
            
            if result['success']:
                result['time'] = elapsed
                self.results.append(result)
                success_count += 1
                print(f"✓ 完成 (ARI={result['ARI']:.4f}, 簇数={result['n_clusters']}, 用时={elapsed:.2f}s)")
            else:
                fail_count += 1
                print(f"✗ 失败: {result['error']}")
        
        print("\n" + "="*80)
        print(f"扫描完成！成功: {success_count}/{total}, 失败: {fail_count}/{total}")
        
        if success_count == 0:
            print("\n警告: 所有测试都失败了！")
            print("建议:")
            print("  1. 检查数据集是否正确")
            print("  2. 调整grid_size范围（可能太小或太大）")
            print("  3. 增大MAX_BUFFER_DISTANCE")
            print("  4. 尝试不同的插值方法")
    
    def find_best_grid_size(self):
        """找到ARI最大的grid_size"""
        if not self.results:
            print("\n" + "="*80)
            print("错误: 没有有效结果")
            print("="*80)
            print("所有测试都失败了，无法找到最佳grid_size")
            return None
        
        # 按ARI排序
        sorted_results = sorted(self.results, key=lambda x: x['ARI'], reverse=True)
        best = sorted_results[0]
        
        print("\n" + "="*80)
        print("最佳结果（ARI最大）")
        print("="*80)
        print(f"最佳 grid_size: {best['grid_size']:.3f}")
        print(f"ARI:           {best['ARI']:.4f}")
        print(f"AMI:           {best['AMI']:.4f}")
        print(f"FMI:           {best['FMI']:.4f}")
        print(f"Silhouette:    {best['Silhouette']:.4f}")
        print(f"发现簇数:      {best['n_clusters']}")
        print(f"噪声点数:      {best['n_noise']}")
        print("="*80)
        
        # 显示前5名
        if len(sorted_results) > 1:
            print("\n前5名结果:")
            print(f"{'排名':<6} {'grid_size':<12} {'ARI':<10} {'簇数':<8} {'噪声':<8}")
            print("-"*50)
            for rank, r in enumerate(sorted_results[:5], 1):
                print(f"{rank:<6} {r['grid_size']:<12.3f} {r['ARI']:<10.4f} "
                      f"{r['n_clusters']:<8} {r['n_noise']:<8}")
        
        return best
    
    def plot_results(self):
        """绘制结果图表"""
        if not self.results:
            print("\n警告: 没有结果可绘制")
            return
        
        grid_sizes = [r['grid_size'] for r in self.results]
        aris = [r['ARI'] for r in self.results]
        amis = [r['AMI'] for r in self.results]
        fmis = [r['FMI'] for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ARI vs grid_size
        axes[0, 0].plot(grid_sizes, aris, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('grid_size')
        axes[0, 0].set_ylabel('ARI')
        axes[0, 0].set_title('ARI vs grid_size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 标记最大值
        max_idx = np.argmax(aris)
        axes[0, 0].plot(grid_sizes[max_idx], aris[max_idx], 'r*', 
                       markersize=15, label=f'Max ARI={aris[max_idx]:.4f}')
        axes[0, 0].legend()
        
        # AMI vs grid_size
        axes[0, 1].plot(grid_sizes, amis, 'g-o', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('grid_size')
        axes[0, 1].set_ylabel('AMI')
        axes[0, 1].set_title('AMI vs grid_size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # FMI vs grid_size
        axes[1, 0].plot(grid_sizes, fmis, 'm-o', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('grid_size')
        axes[1, 0].set_ylabel('FMI')
        axes[1, 0].set_title('FMI vs grid_size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 所有指标对比
        axes[1, 1].plot(grid_sizes, aris, 'b-', linewidth=2, label='ARI')
        axes[1, 1].plot(grid_sizes, amis, 'g-', linewidth=2, label='AMI')
        axes[1, 1].plot(grid_sizes, fmis, 'm-', linewidth=2, label='FMI')
        axes[1, 1].set_xlabel('grid_size')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('所有指标对比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 根据数据集名称生成文件名
        dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
        output_file = f'grid_size_sweep_results_{dataset_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n结果图表已保存为: {output_file}")
        plt.show()
    
    def save_results(self, filename=None, grid_size_start=None, grid_size_end=None, 
                     min_edge=None, original_dim=None):
        """保存结果到文本文件"""
        if filename is None:
            dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
            filename = f'grid_size_sweep_results_{dataset_name}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Grid Size扫描测试结果（支持PCA降维）\n")
            f.write("="*80 + "\n")
            f.write(f"数据集: {DATASET}\n")
            
            if original_dim is not None:
                f.write(f"原始维度: {original_dim}\n")
            
            if self.pca_info:
                f.write(f"PCA降维: 是 ({original_dim}D -> {PCA_N_COMPONENTS}D)\n")
                f.write(f"解释方差比例: {self.pca_info['explained_variance_ratio']}\n")
                f.write(f"累计解释方差: {self.pca_info['cumulative_variance']:.4f}\n")
            else:
                f.write(f"PCA降维: 否\n")
            
            f.write(f"坐标归一化: {'是' if self.normalized else '否'}\n")
            
            if self.normalized and self.scale_info:
                f.write(f"归一化后X范围: [{self.scale_info['x_min']:.2f}, {self.scale_info['x_max']:.2f}]\n")
                f.write(f"归一化后Y范围: [{self.scale_info['y_min']:.2f}, {self.scale_info['y_max']:.2f}]\n")
            
            if min_edge is not None:
                f.write(f"数据最短边: {min_edge:.3f}\n")
            
            if grid_size_start is not None and grid_size_end is not None:
                f.write(f"grid_size范围: {grid_size_start:.3f} - {grid_size_end:.3f}\n")
            
            f.write(f"grid_size步长: {GRID_SIZE_STEP}\n")
            f.write(f"插值方法: {INTERPOLATION_METHOD}\n")
            f.write(f"最大缓冲距离: {MAX_BUFFER_DISTANCE}\n")
            f.write(f"测试数量: {len(self.results)}\n")
            f.write("="*80 + "\n\n")
            
            # 详细结果
            f.write("详细结果\n")
            f.write("-"*80 + "\n")
            f.write(f"{'grid_size':<12} {'簇数':<8} {'噪声':<8} {'ARI':<10} {'AMI':<10} "
                   f"{'FMI':<10} {'Silhouette':<12} {'时间(s)':<10}\n")
            f.write("-"*80 + "\n")
            
            for r in self.results:
                f.write(f"{r['grid_size']:<12.3f} {r['n_clusters']:<8} {r['n_noise']:<8} "
                       f"{r['ARI']:<10.4f} {r['AMI']:<10.4f} {r['FMI']:<10.4f} "
                       f"{r['Silhouette']:<12.4f} {r['time']:<10.2f}\n")
            
            f.write("="*80 + "\n\n")
            
            # 最佳结果
            best = max(self.results, key=lambda x: x['ARI'])
            f.write("最佳结果（ARI最大）\n")
            f.write("-"*80 + "\n")
            f.write(f"最佳 grid_size: {best['grid_size']:.3f}\n")
            f.write(f"ARI:           {best['ARI']:.4f}\n")
            f.write(f"AMI:           {best['AMI']:.4f}\n")
            f.write(f"FMI:           {best['FMI']:.4f}\n")
            f.write(f"Silhouette:    {best['Silhouette']:.4f}\n")
            f.write(f"发现簇数:      {best['n_clusters']}\n")
            f.write(f"噪声点数:      {best['n_noise']}\n")
            f.write("="*80 + "\n")
        
        print(f"\n结果已保存到: {filename}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("="*80)
    print("Grid Size扫描测试（支持PCA降维）")
    print("="*80)
    
    # 读取数据
    print(f"\n正在加载数据集: {DATASET}")
    try:
        X, y_true = load_dataset(DATASET)
        print(f"✓ 数据加载成功")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        exit(1)
    
    original_dim = X.shape[1]
    print(f"\n原始数据信息:")
    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {original_dim}")
    print(f"  真实簇数: {len(np.unique(y_true))}")
    
    # PCA降维（如果需要）
    pca_info = None
    if original_dim > 2 and USE_PCA:
        print(f"\n正在使用PCA降维 ({original_dim}D -> {PCA_N_COMPONENTS}D)...")
        X_pca, pca, explained_variance = apply_pca(X, n_components=PCA_N_COMPONENTS)
        cumulative_variance = np.sum(explained_variance)
        
        print(f"✓ PCA降维完成")
        print(f"  解释方差比例: {explained_variance}")
        print(f"  累计解释方差: {cumulative_variance:.4f} ({cumulative_variance*100:.2f}%)")
        
        pca_info = {
            'pca': pca,
            'explained_variance_ratio': explained_variance,
            'cumulative_variance': cumulative_variance
        }
        
        X = X_pca
    elif original_dim > 2 and not USE_PCA:
        print(f"\n警告: 数据维度为{original_dim}D，但PCA降维已禁用")
        print(f"等值线算法仅支持2D数据，将使用前2个特征")
        X = X[:, :2]
    
    print(f"\n处理后数据信息:")
    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # 计算grid_size范围（在归一化之前）
    grid_size_start, grid_size_end, min_edge = calculate_grid_size_range(X)
    print(f"\n自动计算grid_size范围:")
    print(f"  数据最短边: {min_edge:.3f}")
    print(f"  grid_size起始值: {grid_size_start:.3f} (最短边 / 100)")
    print(f"  grid_size结束值: {grid_size_end:.3f} (最短边 / 2)")
    
    # 归一化坐标
    scale_info = None
    if NORMALIZE_COORDS:
        print(f"\n正在归一化坐标...")
        X_original = X.copy()
        X, scale_info = normalize_coordinates(X)
        print(f"✓ 坐标已归一化到[0, 100]范围")
        print(f"  归一化后X范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
        print(f"  归一化后Y范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
        
        # 归一化后重新计算grid_size范围
        grid_size_start_norm, grid_size_end_norm, min_edge_norm = calculate_grid_size_range(X)
        print(f"\n归一化后grid_size范围:")
        print(f"  数据最短边: {min_edge_norm:.2f}")
        print(f"  grid_size起始值: {grid_size_start_norm:.2f} (最短边 / 100)")
        print(f"  grid_size结束值: {grid_size_end_norm:.2f} (最短边 / 2)")
        
        # 使用归一化后的范围
        grid_size_start = grid_size_start_norm
        grid_size_end = grid_size_end_norm
        min_edge = min_edge_norm
    
    # 创建测试器
    tester = GridSizeTester(X, y_true, normalized=NORMALIZE_COORDS, 
                           scale_info=scale_info, pca_info=pca_info)
    
    # 扫描grid_size
    tester.sweep_grid_size(
        start=grid_size_start,
        end=grid_size_end,
        step=GRID_SIZE_STEP,
        max_buffer_distance=MAX_BUFFER_DISTANCE,
        interpolation_method=INTERPOLATION_METHOD
    )
    
    # 找到最佳grid_size
    best = tester.find_best_grid_size()
    
    if best:
        # 绘制结果
        tester.plot_results()
        
        # 保存结果
        tester.save_results(
            grid_size_start=grid_size_start,
            grid_size_end=grid_size_end,
            min_edge=min_edge,
            original_dim=original_dim
        )
        
        print("\n" + "="*80)
        print("测试完成！")
        print("="*80)
        print(f"推荐使用 grid_size = {best['grid_size']:.3f}")
        print(f"预期 ARI = {best['ARI']:.4f}")
        if pca_info:
            print(f"注意: 结果基于PCA降维后的{PCA_N_COMPONENTS}D数据")
            print(f"      累计解释方差: {pca_info['cumulative_variance']:.4f}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("测试失败")
        print("="*80)
        print("建议:")
        print("  1. 检查数据集路径是否正确")
        print("  2. 调整grid_size范围")
        print("  3. 增大MAX_BUFFER_DISTANCE")
        print("  4. 尝试不同的INTERPOLATION_METHOD")
        if pca_info and pca_info['cumulative_variance'] < 0.8:
            print(f"  5. PCA累计解释方差较低({pca_info['cumulative_variance']:.2%})，可能损失了重要信息")
        print("="*80)
