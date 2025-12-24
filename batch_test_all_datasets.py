"""
批量测试所有数据集
自动遍历data文件夹中的CSV文件，对每个数据集运行多个聚类算法，并保存结果
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi
import time
from datetime import datetime

import ContourBasedClusteringWithoutZ3 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics

# 尝试导入MATLAB集成
try:
    from matlab_integration import MatlabRunner
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("警告: matlab_integration模块未找到，MATLAB算法将被跳过")


# ==================== 配置区域 ====================
DATA_FOLDER = "data"                    # 数据文件夹
OUTPUT_FOLDER = "results"               # 结果输出文件夹
OUTPUT_FILE = "clustering_results.txt"  # 结果文件名

# 算法参数配置
ALGORITHMS_CONFIG = {
    'KMeans': {
        'enabled': True,
        'params': {
            'random_state': 42,
            'n_init': 10
        }
    },
    'DBSCAN': {
        'enabled': True,
        'params': {
            'eps': 1.5,
            'min_samples': 5
        }
    },
    'Contour-Based': {
        'enabled': True,
        'params': {
            'grid_size': None,              # None表示自动计算
            'max_buffer_distance': 5,
            'auto_grid_multiplier': 2.0,
            'optimize_grid': False          # 是否优化grid_size
        }
    },
    'MATLAB-Algorithm_PR': {
        'enabled': True,                    # 设为False可跳过MATLAB算法
        'params': {
            'k': 11,                        # 近邻数量
        }
    }
}

# 是否显示详细输出（设为False可加快运行速度）
VERBOSE = False
# ==================================================


class BatchTester:
    """批量测试器"""
    
    def __init__(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.matlab_runner = None
        
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 初始化MATLAB运行器
        if MATLAB_AVAILABLE and ALGORITHMS_CONFIG['MATLAB-Algorithm_PR']['enabled']:
            try:
                self.matlab_runner = MatlabRunner(method='auto')
                print("MATLAB运行器初始化成功")
            except Exception as e:
                print(f"MATLAB运行器初始化失败: {e}")
                self.matlab_runner = None
    
    def get_dataset_files(self):
        """获取所有CSV数据集文件"""
        files = []
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_folder, filename)
                files.append((filename, filepath))
        return sorted(files)
    
    def load_dataset(self, filepath):
        """加载数据集"""
        df = pd.read_csv(filepath)
        X = df[['x', 'y']].values
        y_true = df['target'].values
        return X, y_true
    
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
    
    def calculate_auto_grid_size(self, coords, grid_multiplier=2.0):
        """自动计算grid_size"""
        n_points = len(coords)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        target_grid_count = n_points * grid_multiplier
        grid_size = np.sqrt(area / target_grid_count)
        return round(grid_size, 3)
    
    def run_kmeans(self, X, y_true, n_clusters, params):
        """运行KMeans"""
        try:
            kmeans = KMeans(n_clusters=n_clusters, **params)
            labels = kmeans.fit_predict(X)
            
            inertia = self.calculate_inertia(X, labels)
            silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
            metrics = ClusteringMetrics.evaluate_all(y_true, labels)
            
            return {
                'success': True,
                'labels': labels,
                'n_clusters': len(set(labels)),
                'inertia': inertia,
                'silhouette': silhouette,
                'AMI': metrics['AMI'],
                'ARI': metrics['ARI'],
                'FMI': metrics['FMI']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_dbscan(self, X, y_true, params):
        """运行DBSCAN"""
        try:
            dbscan = DBSCAN(**params)
            labels = dbscan.fit_predict(X)
            
            inertia = self.calculate_inertia(X, labels)
            
            if len(set(labels)) > 1:
                mask = labels != -1
                if np.sum(mask) > 0 and len(np.unique(labels[mask])) > 1:
                    silhouette = silhouette_score(X[mask], labels[mask])
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            metrics = ClusteringMetrics.evaluate_all(y_true, labels)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            return {
                'success': True,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'inertia': inertia,
                'silhouette': silhouette,
                'AMI': metrics['AMI'],
                'ARI': metrics['ARI'],
                'FMI': metrics['FMI']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_contour(self, X, y_true, params):
        """运行Contour-Based聚类"""
        try:
            # 计算grid_size
            if params.get('optimize_grid', False):
                from optimize_grid_size import GridSizeOptimizer
                optimizer = GridSizeOptimizer(X, y_true, params['max_buffer_distance'])
                grid_size, _, _ = optimizer.optimize(
                    initial_step=None,
                    min_step=0.01,
                    step_decay=0.5,
                    max_iterations=15,
                    tolerance=0.001,
                    min_grid_size=0.1,
                    verbose=False
                )
            elif params['grid_size'] is None:
                grid_size = self.calculate_auto_grid_size(X, params['auto_grid_multiplier'])
            else:
                grid_size = params['grid_size']
            
            # 计算密度
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
            
            # 准备数据
            combined_data = [(*point, density[i], i) for i, point in enumerate(X)]
            
            # 运行聚类（禁用可视化）
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
                combined_data,
                grid_size=grid_size,
                max_buffer_distance=params['max_buffer_distance']
            )
            
            # 临时禁用可视化
            original_generate = cluster.generate_virtual_points
            def silent_generate():
                # 简化版本，不显示交互窗口
                if not cluster.actual_points:
                    return []
                
                x_coords = [p[0] for p in cluster.actual_points]
                y_coords = [p[1] for p in cluster.actual_points]
                z_vals = [p[2] for p in cluster.actual_points]
                avg_z = np.mean(z_vals)
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                i_start = min_x - 2 * cluster.grid_size
                i_end = max_x + 2 * cluster.grid_size
                j_start = min_y - 2 * cluster.grid_size
                j_end = max_y + 2 * cluster.grid_size
                
                virtual_points = []
                i = i_start
                while i < i_end:
                    j = j_start
                    while j < j_end:
                        has_point = any(
                            i <= x < i + cluster.grid_size and
                            j <= y < j + cluster.grid_size
                            for (x, y, _, _) in cluster.actual_points
                        )
                        if not has_point:
                            virtual_points.append((i + cluster.grid_size / 2,
                                                 j + cluster.grid_size / 2,
                                                 0.0, -1))
                        j += cluster.grid_size
                    i += cluster.grid_size
                
                cluster.virtual_points = virtual_points
                
                x_ticks = np.arange(i_start, i_end + cluster.grid_size, cluster.grid_size)
                y_ticks = np.arange(j_start, j_end + cluster.grid_size, cluster.grid_size)
                
                return virtual_points, x_ticks, y_ticks
            
            cluster.generate_virtual_points = silent_generate
            
            # 执行聚类
            cluster.perform_clustering()
            labels = cluster.get_predicted_labels()
            
            # 计算指标
            inertia = self.calculate_inertia(X, labels)
            
            if len(set(labels)) > 1:
                mask = np.array(labels) != -1
                if np.sum(mask) > 0 and len(np.unique(np.array(labels)[mask])) > 1:
                    silhouette = silhouette_score(X[mask], np.array(labels)[mask])
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            metrics = ClusteringMetrics.evaluate_all(y_true, labels)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            return {
                'success': True,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'grid_size': grid_size,
                'inertia': inertia,
                'silhouette': silhouette,
                'AMI': metrics['AMI'],
                'ARI': metrics['ARI'],
                'FMI': metrics['FMI']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_matlab_algorithm_pr(self, X, y_true, n_clusters, params):
        """运行MATLAB Algorithm_PR"""
        if not self.matlab_runner:
            return {'success': False, 'error': 'MATLAB运行器未初始化'}
        
        try:
            # 准备数据 [x, y, label]
            data_with_labels = np.column_stack([X, y_true])
            
            # 运行算法
            labels, matlab_metrics = self.matlab_runner.run_algorithm_pr(
                data_with_labels, n_clusters, params['k']
            )
            
            if labels is None:
                return {'success': False, 'error': 'MATLAB算法执行失败'}
            
            # 计算额外指标
            inertia = self.calculate_inertia(X, labels)
            
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = -1
            
            n_clusters_found = len(set(labels))
            
            return {
                'success': True,
                'labels': labels,
                'n_clusters': n_clusters_found,
                'inertia': inertia,
                'silhouette': silhouette,
                'AMI': matlab_metrics['AMI'],
                'ARI': matlab_metrics['ARI'],
                'FMI': matlab_metrics['FMI']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_dataset(self, filename, filepath):
        """测试单个数据集"""
        print(f"\n{'='*80}")
        print(f"测试数据集: {filename}")
        print(f"{'='*80}")
        
        # 加载数据
        X, y_true = self.load_dataset(filepath)
        n_true_clusters = len(np.unique(y_true))
        
        print(f"样本数: {len(X)}, 真实簇数: {n_true_clusters}")
        
        results = {
            'filename': filename,
            'n_samples': len(X),
            'n_true_clusters': n_true_clusters,
            'algorithms': {}
        }
        
        # 测试KMeans
        if ALGORITHMS_CONFIG['KMeans']['enabled']:
            print("  运行 KMeans...", end=' ')
            start_time = time.time()
            result = self.run_kmeans(X, y_true, n_true_clusters, 
                                    ALGORITHMS_CONFIG['KMeans']['params'])
            elapsed = time.time() - start_time
            result['time'] = elapsed
            results['algorithms']['KMeans'] = result
            if result['success']:
                print(f"完成 (ARI={result['ARI']:.3f}, 用时={elapsed:.2f}s)")
            else:
                print(f"失败: {result['error']}")
        
        # 测试DBSCAN
        if ALGORITHMS_CONFIG['DBSCAN']['enabled']:
            print("  运行 DBSCAN...", end=' ')
            start_time = time.time()
            result = self.run_dbscan(X, y_true, 
                                    ALGORITHMS_CONFIG['DBSCAN']['params'])
            elapsed = time.time() - start_time
            result['time'] = elapsed
            results['algorithms']['DBSCAN'] = result
            if result['success']:
                print(f"完成 (ARI={result['ARI']:.3f}, 用时={elapsed:.2f}s)")
            else:
                print(f"失败: {result['error']}")
        
        # 测试Contour-Based
        if ALGORITHMS_CONFIG['Contour-Based']['enabled']:
            print("  运行 Contour-Based...", end=' ')
            start_time = time.time()
            result = self.run_contour(X, y_true, 
                                     ALGORITHMS_CONFIG['Contour-Based']['params'])
            elapsed = time.time() - start_time
            result['time'] = elapsed
            results['algorithms']['Contour-Based'] = result
            if result['success']:
                print(f"完成 (ARI={result['ARI']:.3f}, grid_size={result['grid_size']:.3f}, 用时={elapsed:.2f}s)")
            else:
                print(f"失败: {result['error']}")
        
        # 测试MATLAB Algorithm_PR
        if ALGORITHMS_CONFIG['MATLAB-Algorithm_PR']['enabled'] and MATLAB_AVAILABLE:
            print("  运行 MATLAB Algorithm_PR...", end=' ')
            start_time = time.time()
            result = self.run_matlab_algorithm_pr(X, y_true, n_true_clusters,
                                                  ALGORITHMS_CONFIG['MATLAB-Algorithm_PR']['params'])
            elapsed = time.time() - start_time
            result['time'] = elapsed
            results['algorithms']['MATLAB-Algorithm_PR'] = result
            if result['success']:
                print(f"完成 (ARI={result['ARI']:.3f}, 用时={elapsed:.2f}s)")
            else:
                print(f"失败: {result['error']}")
        
        return results
    
    def save_results(self, all_results, output_file):
        """保存结果到文本文件"""
        filepath = os.path.join(self.output_folder, output_file)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 写入标题
            f.write("="*100 + "\n")
            f.write("聚类算法批量测试结果\n")
            f.write("="*100 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集数量: {len(all_results)}\n")
            f.write(f"算法: {', '.join([k for k, v in ALGORITHMS_CONFIG.items() if v['enabled']])}\n")
            f.write("="*100 + "\n\n")
            
            # 写入每个数据集的详细结果
            for result in all_results:
                f.write(f"\n{'='*100}\n")
                f.write(f"数据集: {result['filename']}\n")
                f.write(f"{'='*100}\n")
                f.write(f"样本数: {result['n_samples']}\n")
                f.write(f"真实簇数: {result['n_true_clusters']}\n")
                f.write(f"{'-'*100}\n")
                
                for algo_name, algo_result in result['algorithms'].items():
                    f.write(f"\n算法: {algo_name}\n")
                    f.write(f"{'-'*50}\n")
                    
                    if algo_result['success']:
                        f.write(f"  发现簇数: {algo_result['n_clusters']}\n")
                        if 'n_noise' in algo_result:
                            f.write(f"  噪声点数: {algo_result['n_noise']}\n")
                        if 'grid_size' in algo_result:
                            f.write(f"  grid_size: {algo_result['grid_size']:.3f}\n")
                        f.write(f"  AMI:        {algo_result['AMI']:.4f}\n")
                        f.write(f"  ARI:        {algo_result['ARI']:.4f}\n")
                        f.write(f"  FMI:        {algo_result['FMI']:.4f}\n")
                        f.write(f"  Silhouette: {algo_result['silhouette']:.4f}\n")
                        f.write(f"  Inertia:    {algo_result['inertia']:.2f}\n")
                        f.write(f"  运行时间:   {algo_result['time']:.2f}s\n")
                    else:
                        f.write(f"  失败: {algo_result['error']}\n")
                
                f.write(f"\n")
            
            # 写入汇总表格
            f.write(f"\n{'='*100}\n")
            f.write("汇总对比表\n")
            f.write(f"{'='*100}\n\n")
            
            # 表头
            f.write(f"{'数据集':<20} {'算法':<20} {'ARI':<10} {'AMI':<10} {'FMI':<10} "
                   f"{'Silhouette':<12} {'时间(s)':<10}\n")
            f.write(f"{'-'*100}\n")
            
            # 数据行
            for result in all_results:
                for algo_name, algo_result in result['algorithms'].items():
                    if algo_result['success']:
                        f.write(f"{result['filename']:<20} {algo_name:<20} "
                               f"{algo_result['ARI']:<10.4f} {algo_result['AMI']:<10.4f} "
                               f"{algo_result['FMI']:<10.4f} {algo_result['silhouette']:<12.4f} "
                               f"{algo_result['time']:<10.2f}\n")
                    else:
                        f.write(f"{result['filename']:<20} {algo_name:<20} {'FAILED':<10} "
                               f"{'FAILED':<10} {'FAILED':<10} {'FAILED':<12} "
                               f"{algo_result.get('time', 0):<10.2f}\n")
            
            f.write(f"{'='*100}\n")
            
            # 计算平均值
            f.write(f"\n平均性能指标\n")
            f.write(f"{'-'*100}\n")
            
            for algo_name in [k for k, v in ALGORITHMS_CONFIG.items() if v['enabled']]:
                ari_values = []
                ami_values = []
                fmi_values = []
                time_values = []
                
                for result in all_results:
                    if algo_name in result['algorithms']:
                        algo_result = result['algorithms'][algo_name]
                        if algo_result['success']:
                            ari_values.append(algo_result['ARI'])
                            ami_values.append(algo_result['AMI'])
                            fmi_values.append(algo_result['FMI'])
                            time_values.append(algo_result['time'])
                
                if ari_values:
                    f.write(f"\n{algo_name}:\n")
                    f.write(f"  平均 ARI: {np.mean(ari_values):.4f} (±{np.std(ari_values):.4f})\n")
                    f.write(f"  平均 AMI: {np.mean(ami_values):.4f} (±{np.std(ami_values):.4f})\n")
                    f.write(f"  平均 FMI: {np.mean(fmi_values):.4f} (±{np.std(fmi_values):.4f})\n")
                    f.write(f"  平均时间: {np.mean(time_values):.2f}s (±{np.std(time_values):.2f}s)\n")
            
            f.write(f"\n{'='*100}\n")
        
        print(f"\n结果已保存到: {filepath}")
    
    def run_batch_test(self):
        """运行批量测试"""
        print("="*80)
        print("开始批量测试")
        print("="*80)
        
        # 获取所有数据集
        datasets = self.get_dataset_files()
        print(f"找到 {len(datasets)} 个数据集")
        
        # 测试每个数据集
        all_results = []
        for i, (filename, filepath) in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}]", end=' ')
            result = self.test_dataset(filename, filepath)
            all_results.append(result)
        
        # 保存结果
        print(f"\n{'='*80}")
        print("保存结果...")
        self.save_results(all_results, OUTPUT_FILE)
        
        print(f"{'='*80}")
        print("批量测试完成！")
        print(f"{'='*80}")
        
        # 关闭MATLAB运行器
        if self.matlab_runner:
            try:
                self.matlab_runner.close()
                print("MATLAB运行器已关闭")
            except:
                pass
        
        return all_results


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 创建测试器
    tester = BatchTester(DATA_FOLDER, OUTPUT_FOLDER)
    
    # 运行批量测试
    results = tester.run_batch_test()
    
    print(f"\n测试完成！共测试 {len(results)} 个数据集")
    print(f"结果保存在: {os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)}")
