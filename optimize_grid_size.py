"""
基于梯度下降的grid_size自动优化
使用ARI作为评价标准
"""

import numpy as np
from scipy.spatial import Voronoi
import ContourBasedClusteringWithoutZ3 as ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics
from sklearn.metrics import silhouette_score, adjusted_rand_score

class GridSizeOptimizer:
    """grid_size优化器"""
    
    def __init__(self, coords, true_labels, max_buffer_distance=5):
        """
        初始化优化器
        
        参数:
            coords: 坐标数组 shape=(n, 2)
            true_labels: 真实标签
            max_buffer_distance: 最大缓冲距离
        """
        self.coords = coords
        self.true_labels = true_labels
        self.max_buffer_distance = max_buffer_distance
        
        # 计算初始grid_size（格子数=点数×2）
        self.initial_grid_size = self._calculate_initial_grid_size()*2
        
        # 优化历史
        self.history = []
    
    def _calculate_initial_grid_size(self, multiplier=2.0):
        """计算初始grid_size"""
        n_points = len(self.coords)
        x_min, y_min = self.coords.min(axis=0)
        x_max, y_max = self.coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        
        target_grid_count = n_points * multiplier
        grid_size = np.sqrt(area / target_grid_count)
        
        # 保留3位小数
        return round(grid_size, 3)
    
    def _run_clustering(self, grid_size):
        #保留3位小数

        """
        运行聚类并返回ARI
        
        参数:
            grid_size: 网格大小
            
        返回:
            ari: ARI值
            labels: 聚类标签
        """
        try:
            # 计算密度
            points = self.coords
            vor = Voronoi(points)
            density = np.zeros(len(points))
            
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
            combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
            
            # 运行聚类
            cluster = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
                combined_data,
                grid_size=grid_size,
                max_buffer_distance=self.max_buffer_distance
            )
            
            cluster.perform_clustering()
            labels = cluster.get_predicted_labels()
            
            # 计算ARI
            ari = adjusted_rand_score(self.true_labels, labels)
            
            return ari, labels
            
        except Exception as e:
            print(f"  聚类失败 (grid_size={grid_size:.3f}): {e}")
            return -1.0, None
    
    def optimize(self, 
                 initial_step=None,
                 min_step=0.01,
                 step_decay=0.5,
                 max_iterations=20,
                 tolerance=0.001,
                 min_grid_size=0.1,
                 verbose=True):
        """
        优化grid_size
        
        参数:
            initial_step: 初始步长（默认为initial_grid_size的10%）
            min_step: 最小步长
            step_decay: 步长衰减率（当ARI下降时）
            max_iterations: 最大迭代次数
            tolerance: ARI改善的最小阈值
            min_grid_size: grid_size的最小值
            verbose: 是否打印详细信息
            
        返回:
            best_grid_size: 最优grid_size
            best_ari: 最优ARI
            best_labels: 最优聚类标签
        """
        if verbose:
            print("=" * 80)
            print("开始优化grid_size")
            print("=" * 80)
            print(f"初始grid_size: {self.initial_grid_size:.3f}")
            print(f"最大迭代次数: {max_iterations}")
            print(f"最小步长: {min_step}")
            print(f"步长衰减率: {step_decay}")
            print(f"容忍度: {tolerance}")
            print("-" * 80)
        
        # 初始化
        current_grid_size = self.initial_grid_size
        if initial_step is None:
            current_step = self.initial_grid_size * 0.1  # 初始步长为10%
        else:
            current_step = initial_step
        
        # 评估初始grid_size
        if verbose:
            print(f"\n迭代 0: 评估初始grid_size")
        current_ari, current_labels = self._run_clustering(current_grid_size)
        
        if verbose:
            print(f"  grid_size: {current_grid_size:.3f}")
            print(f"  ARI: {current_ari:.4f}")
        
        # 记录最优结果
        best_grid_size = current_grid_size
        best_ari = current_ari
        best_labels = current_labels
        
        self.history.append({
            'iteration': 0,
            'grid_size': current_grid_size,
            'ari': current_ari,
            'step': current_step,
            'action': 'initial'
        })
        
        # 迭代优化
        no_improvement_count = 0
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n迭代 {iteration}:")
            
            # 尝试减小grid_size
            new_grid_size = max(current_grid_size - current_step, min_grid_size)
            # 保留3位小数
            new_grid_size = round(new_grid_size, 3)
            
            if new_grid_size == min_grid_size and new_grid_size == current_grid_size:
                if verbose:
                    print(f"  已达到最小grid_size限制: {min_grid_size}")
                break
            
            if verbose:
                print(f"  尝试 grid_size: {new_grid_size:.3f} (步长: {current_step:.4f})")
            
            # 评估新的grid_size
            new_ari, new_labels = self._run_clustering(new_grid_size)
            
            if verbose:
                print(f"  ARI: {new_ari:.4f} (变化: {new_ari - current_ari:+.4f})")
            
            # 判断是否改善
            improvement = new_ari - current_ari
            
            if improvement > tolerance:
                # ARI提升，接受新的grid_size
                if verbose:
                    print(f"  ✓ ARI提升，接受新grid_size")
                
                current_grid_size = new_grid_size
                current_ari = new_ari
                current_labels = new_labels
                
                # 更新最优结果
                if new_ari > best_ari:
                    best_grid_size = new_grid_size
                    best_ari = new_ari
                    best_labels = new_labels
                
                self.history.append({
                    'iteration': iteration,
                    'grid_size': new_grid_size,
                    'ari': new_ari,
                    'step': current_step,
                    'action': 'accept'
                })
                
                no_improvement_count = 0
                
            elif improvement > -tolerance:
                # ARI变化很小，认为收敛
                if verbose:
                    print(f"  → ARI变化很小，认为已收敛")
                
                self.history.append({
                    'iteration': iteration,
                    'grid_size': new_grid_size,
                    'ari': new_ari,
                    'step': current_step,
                    'action': 'converged'
                })
                
                no_improvement_count += 1
                if no_improvement_count >= 2:
                    if verbose:
                        print(f"  停止优化：连续{no_improvement_count}次无明显改善")
                    break
                
            else:
                # ARI下降，减小步长并回退
                if verbose:
                    print(f"  ✗ ARI下降，减小步长并回退")
                
                current_step *= step_decay
                
                self.history.append({
                    'iteration': iteration,
                    'grid_size': new_grid_size,
                    'ari': new_ari,
                    'step': current_step,
                    'action': 'reject_and_decay'
                })
                
                if current_step < min_step:
                    if verbose:
                        print(f"  步长过小 ({current_step:.4f} < {min_step})，停止优化")
                    break
                
                no_improvement_count += 1
                if no_improvement_count >= 10:
                    if verbose:
                        print(f"  停止优化：连续{no_improvement_count}次无改善")
                    break
        
        # 输出最终结果
        if verbose:
            print("\n" + "=" * 80)
            print("优化完成")
            print("=" * 80)
            print(f"初始 grid_size: {self.initial_grid_size:.3f}, ARI: {self.history[0]['ari']:.4f}")
            print(f"最优 grid_size: {best_grid_size:.3f}, ARI: {best_ari:.4f}")
            print(f"ARI提升: {best_ari - self.history[0]['ari']:+.4f}")
            print(f"总迭代次数: {len(self.history) - 1}")
            print("=" * 80)
        
        return best_grid_size, best_ari, best_labels
    
    def plot_history(self):
        """绘制优化历史"""
        import matplotlib.pyplot as plt
        
        if not self.history:
            print("没有优化历史")
            return
        
        iterations = [h['iteration'] for h in self.history]
        grid_sizes = [h['grid_size'] for h in self.history]
        aris = [h['ari'] for h in self.history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制grid_size变化
        ax1.plot(iterations, grid_sizes, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('grid_size')
        ax1.set_title('grid_size优化过程')
        ax1.grid(True, alpha=0.3)
        
        # 标记最优点
        best_idx = np.argmax(aris)
        ax1.plot(iterations[best_idx], grid_sizes[best_idx], 'r*', 
                markersize=15, label='最优点')
        ax1.legend()
        
        # 绘制ARI变化
        ax2.plot(iterations, aris, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('ARI')
        ax2.set_title('ARI变化')
        ax2.grid(True, alpha=0.3)
        
        # 标记最优点
        ax2.plot(iterations[best_idx], aris[best_idx], 'r*', 
                markersize=15, label='最优点')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_history(self):
        """打印优化历史"""
        print("\n" + "=" * 100)
        print("优化历史详情")
        print("=" * 100)
        print(f"{'迭代':<8} {'grid_size':<12} {'ARI':<10} {'步长':<10} {'动作':<20}")
        print("-" * 100)
        
        for h in self.history:
            action_str = {
                'initial': '初始评估',
                'accept': '✓ 接受',
                'reject_and_decay': '✗ 拒绝并减小步长',
                'converged': '→ 收敛'
            }.get(h['action'], h['action'])
            
            print(f"{h['iteration']:<8} {h['grid_size']:<12.3f} {h['ari']:<10.4f} "
                  f"{h['step']:<10.4f} {action_str:<20}")
        
        print("=" * 100)


# 使用示例
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print("grid_size自动优化示例\n")
    
    # 读取数据
    dataset = "data/jain.csv"
    df = pd.read_csv(dataset)
    X = df[['x', 'y']].values
    y_true = df['target'].values
    
    print(f"数据集: {dataset}")
    print(f"样本数: {len(X)}")
    print(f"真实簇数: {len(np.unique(y_true))}\n")
    
    # 创建优化器
    optimizer = GridSizeOptimizer(X, y_true, max_buffer_distance=5)
    
    # 运行优化
    best_grid_size, best_ari, best_labels = optimizer.optimize(
        initial_step=None,      # 自动设置为初始grid_size的10%
        min_step=0.001,          # 最小步长
        step_decay=0.5,         # 步长衰减率
        max_iterations=20,      # 最大迭代次数
        tolerance=0.001,        # ARI改善阈值
        min_grid_size=0.1,      # grid_size最小值
        verbose=True            # 打印详细信息
    )
    
    # 打印历史
    optimizer.print_history()
    
    # 绘制优化过程
    optimizer.plot_history()
    
    print(f"\n推荐使用的grid_size: {best_grid_size:.3f}")
    print(f"对应的ARI: {best_ari:.4f}")
