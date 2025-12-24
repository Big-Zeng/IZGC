"""
ContourBasedClusteringWithoutZ4 - 支持多种插值方法的等值线聚类
支持的插值方法：
- linear: 线性插值（默认）
- cubic: 双三次样条插值
- idw: 反距离加权插值（Inverse Distance Weighting）
"""

import math
import numpy as np
from matplotlib.contour import ContourSet
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon, LineString
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import ndimage
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree

plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题


class ContourBasedClusteringWithoutZ:
    def __init__(self, actual_points, grid_size=1.0, max_buffer_distance=5, 
                     interpolation_method='linear', idw_power=3, idw_neighbors=10,
                 cubic_smoothing=0.0):
        """
        初始化聚类算法

        参数:
        actual_points -- 实际点列表，格式为[(x, y, z, clientId), ...]
        grid_size -- 生成虚拟点的网格大小 (默认1.0)
        max_buffer_distance -- 最大缓冲区距离 (默认5)
        interpolation_method -- 插值方法: 'linear', 'cubic', 'idw' (默认'linear')
        idw_power -- IDW插值的幂次 (默认2.0)
        idw_neighbors -- IDW插值使用的近邻点数 (默认8)
        cubic_smoothing -- 三次插值的平滑参数 (默认0.0)
        """
        self.actual_points = actual_points
        self.grid_size = grid_size
        self.max_buffer_distance = int(max_buffer_distance)
        self.interpolation_method = interpolation_method
        self.idw_power = idw_power
        self.idw_neighbors = idw_neighbors
        self.cubic_smoothing = cubic_smoothing
        
        self.virtual_points = []
        self.contour_areas = []
        self.density_centers = []
        self.point_to_density_center = {}
        self.manual_points = []
        
        # 内部类定义
        class ContourArea:
            def __init__(self, polygon, points):
                self.polygon = polygon
                self.points = points
                self.cargoVol = self.caculate_point_vol()
                self.isLeave = True
                self.parent = None
                self.childLength = 0
                self.center = self.get_centroid()

            def caculate_point_vol(self):
                """计算区域内的货物总量"""
                return sum(point[2] for point in self.points)

            def get_centroid(self):
                """计算区域的中心点"""
                x_sum = sum(point[0] for point in self.points)
                y_sum = sum(point[1] for point in self.points)
                return (round(x_sum / len(self.points), 2), round(y_sum / len(self.points), 2))

        self.ContourArea = ContourArea

    def generate_virtual_points(self):
        """生成虚拟坐标点，低货量点周围细化网格"""
        if not self.actual_points:
            return []

        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        z_vals = [p[2] for p in self.actual_points]
        avg_z = np.mean(z_vals)

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 扩展边界
        i_start = min_x - 2 * self.grid_size
        i_end = max_x + 2 * self.grid_size
        j_start = min_y - 2 * self.grid_size
        j_end = max_y + 2 * self.grid_size

        virtual_points = []

        # 正常网格部分
        i = i_start
        while i < i_end:
            j = j_start
            while j < j_end:
                has_point = any(
                    i <= x < i + self.grid_size and
                    j <= y < j + self.grid_size
                    for (x, y, _, _) in self.actual_points
                )
                if not has_point:
                    virtual_points.append((i + self.grid_size / 2,
                                           j + self.grid_size / 2,
                                           0.0, -1))
                j += self.grid_size
            i += self.grid_size

        # 低货量区域加密网格
        fine_grid_size = self.grid_size / 2
        for p in self.actual_points:
            if p[2] < avg_z:
                px, py = p[0], p[1]
                local_x_start = px - self.grid_size
                local_x_end = px + self.grid_size
                local_y_start = py - self.grid_size
                local_y_end = py + self.grid_size

                xi = local_x_start
                while xi < local_x_end:
                    yj = local_y_start
                    while yj < local_y_end:
                        has_point = any(
                            xi <= x < xi + fine_grid_size and
                            yj <= y < yj + fine_grid_size
                            for (x, y, _, _) in self.actual_points
                        )
                        if (not has_point) and not any(
                                abs(vx - (xi + fine_grid_size / 2)) < 1e-6 and
                                abs(vy - (yj + fine_grid_size / 2)) < 1e-6
                                for (vx, vy, _, _) in virtual_points
                        ):
                            virtual_points.append(
                                (xi + fine_grid_size / 2, yj + fine_grid_size / 2, 0.0, -1)
                            )
                        yj += fine_grid_size
                    xi += fine_grid_size

        # 绘制交互界面
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        print(f"网格总数: {(len(x_ticks) - 1) * (len(y_ticks) - 1)}, 自动虚拟点数: {len(virtual_points)}")
        print(f"低货量点平均阈值: {avg_z:.3f}")
        print(f"插值方法: {self.interpolation_method}")

        fig, ax = plt.subplots(figsize=(10, 8))
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        ax.scatter(x_coords, y_coords, c='red', s=50, label="实际点")
        if virtual_points:
            vx = [p[0] for p in virtual_points]
            vy = [p[1] for p in virtual_points]
            ax.scatter(vx, vy, c='blue', s=30, marker='x', label="自动虚拟点")

        ax.set_title(f"网格显示 (插值方法: {self.interpolation_method})\n"
                    f"低货量点已局部细化。\n鼠标左键添加额外零值点，关闭窗口继续。", 
                    color='blue')

        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                new_x, new_y = event.xdata, event.ydata
                new_point = (new_x, new_y, 0.0, -2)
                self.virtual_points.append(new_point)
                self.manual_points.append(new_point)
                ax.scatter(new_x, new_y, c='green', s=60, marker='+', label="_nolegend_")
                ax.text(new_x, new_y, " 0", fontsize=8, color='green')
                fig.canvas.draw()
                print(f"手动添加零值点: ({new_x:.2f}, {new_y:.2f})")

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.legend(loc='upper right')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        self.virtual_points = virtual_points + self.manual_points

        print(f"交互结束。自动点: {len(virtual_points)}, 手动点: {len(self.manual_points)}")
        return self.virtual_points, x_ticks, y_ticks

    @staticmethod
    def calculate_distance(point1, point2):
        """计算两点之间的欧氏距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def is_point_in_region(x, y, polygon):
        """判断点是否在多边形区域内"""
        point = Point(x, y)
        poly = Polygon(polygon)
        return poly.contains(point)

    def line_to_polygon(self, contour_set):
        """将等值线转换为多边形区域"""
        face_polygons = []
        for path in contour_set.get_paths():
            paths = path.to_polygons()
            if paths:
                for path in paths:
                    face_polygons.append(path)
        return face_polygons

    def generate_contour_areas(self):
        """生成等值线区域并建立层次结构（支持多种插值方法）"""
        # 生成虚拟点
        virtual_points, x_ticks, y_ticks = self.generate_virtual_points()

        all_points = self.actual_points + self.virtual_points

        # 准备插值数据
        x1 = np.array([p[0] for p in all_points])
        y1 = np.array([p[1] for p in all_points])
        z1 = np.array([p[2] for p in all_points])

        X, Y = np.meshgrid(x_ticks, y_ticks)

        # 根据不同插值方法进行插值
        print(f"使用 {self.interpolation_method} 插值方法...")
        
        if self.interpolation_method == 'idw':
            # IDW (Inverse Distance Weighting) 插值
            print(f"  IDW参数: power={self.idw_power}, neighbors={self.idw_neighbors}")
            tree = cKDTree(np.c_[x1, y1])
            distances, indices = tree.query(np.c_[X.flat, Y.flat], k=self.idw_neighbors)
            
            # 避免除零错误
            distances = np.where(distances == 0, 1e-10, distances)
            
            weights = 1.0 / distances**self.idw_power
            zi = np.sum(weights * z1[indices], axis=1) / np.sum(weights, axis=1)
            zi = zi.reshape(X.shape)
            
        elif self.interpolation_method == 'cubic':
            # 双三次样条插值
            print(f"  Cubic参数: smoothing={self.cubic_smoothing}")
            zi = griddata((x1, y1), z1, (X, Y), method='cubic')
            
            # 处理NaN值
            zi = np.nan_to_num(zi, nan=0)
            
            # 添加平滑处理
            if self.cubic_smoothing > 0:
                zi = ndimage.gaussian_filter(zi, sigma=self.cubic_smoothing)
                print(f"  应用高斯平滑，sigma={self.cubic_smoothing}")
                
        else:
            # 默认使用线性插值或其他scipy支持的方法
            zi = griddata((x1, y1), z1, (X, Y), method=self.interpolation_method)
            zi = np.nan_to_num(zi, nan=0)

        # 生成等值线
        fig, ax = plt.subplots(figsize=(10, 8))
        contour_set = ax.contour(X, Y, zi, levels=1, colors='black')
        
        # 展示等值线图
        ax.set_title(f'等值线图 (插值方法: {self.interpolation_method})')

        # 绘制实际点
        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        ax.scatter(x_coords, y_coords, c='red', s=50, label="实际点", zorder=5)
        
        # 在每个实际点旁边显示Z值
        for i, point in enumerate(self.actual_points):
            ax.text(point[0], point[1], f' {point[2]:.1f}',
                    fontsize=8, color='darkred',
                    verticalalignment='bottom',
                    horizontalalignment='left')

        # 绘制虚拟点
        if self.virtual_points:
            vx = [p[0] for p in self.virtual_points]
            vy = [p[1] for p in self.virtual_points]
            ax.scatter(vx, vy, c='blue', s=30, marker='x', label="虚拟点", alpha=0.5)
        
        ax.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(ax.contourf(X, Y, zi, levels=10, alpha=0.3), ax=ax, label='密度值')
        plt.show()

        # 转换为多边形区域
        face_polygons = self.line_to_polygon(contour_set)

        # 创建等值线区域对象
        contour_areas = []
        for polygon in face_polygons:
            points_in_poly = []
            for point in self.actual_points:
                if self.is_point_in_region(point[0], point[1], polygon):
                    points_in_poly.append(point)
            if points_in_poly:
                contour_areas.append(self.ContourArea(polygon, points_in_poly))

        self.contour_areas = contour_areas

        # 建立区域层次结构
        for area in self.contour_areas:
            area_poly = Polygon(area.polygon)
            for other in self.contour_areas:
                if area == other:
                    continue
                other_poly = Polygon(other.polygon)
                if area_poly.within(other_poly):
                    area.parent = other
                    other.isLeave = False
                    area.childLength += 1

        return self.contour_areas

    def get_density_centers(self):
        """获取密度中心点（关键聚类中心）"""
        outCircle = [area for area in self.contour_areas if area.parent is None]
        self.density_centers = outCircle
        return outCircle

    def assign_points_to_centers(self):
        """通过缓冲区扩展将点分配给密度中心"""
        if not self.density_centers:
            return {}

        point_assignments = {}
        assigned_points = set()

        for buffer_dist in range(0, self.max_buffer_distance + 1):
            current_assignments = {}

            for center in self.density_centers:
                center_poly = Polygon(center.polygon)
                buffer_zone = center_poly.buffer(buffer_dist)

                for point in self.actual_points:
                    if point in assigned_points:
                        continue

                    pt = Point(point[0], point[1])
                    if buffer_zone.contains(pt):
                        if point not in current_assignments:
                            current_assignments[point] = []
                        current_assignments[point].append(center)

            for point, centers in current_assignments.items():
                if len(centers) > 1:
                    min_dist = float('inf')
                    closest_center = None
                    for center in centers:
                        dist = self.calculate_distance(point, center.center)
                        if dist < min_dist:
                            min_dist = dist
                            closest_center = center
                    current_assignments[point] = [closest_center]

                self.point_to_density_center[point] = current_assignments[point]
                assigned_points.add(point)

        return self.point_to_density_center

    def visualize_clusters(self):
        """可视化聚类结果"""
        if not self.density_centers:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

        for i, center in enumerate(self.density_centers):
            color = colors[i % len(colors)]

            poly_patch = patches.Polygon(
                center.polygon,
                closed=True,
                fill=None,
                edgecolor=color,
                linestyle='dashed'
            )
            ax.add_patch(poly_patch)

            for point, centers in self.point_to_density_center.items():
                if center in centers:
                    ax.plot(point[0], point[1], 'o', color=color, markersize=6)

        all_assigned = {point for point in self.point_to_density_center}
        for point in self.actual_points:
            if point not in all_assigned:
                ax.plot(point[0], point[1], 'ko', markersize=6)

        ax.set_title(f'聚类结果 (插值方法: {self.interpolation_method})')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def calculate_inertia(self, X, labels):
        """计算簇内误差平方和（Inertia）"""
        n_clusters = len(np.unique(labels))
        inertia = 0.0

        for k in range(n_clusters):
            cluster_points = X[labels == k]
            cluster_center = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            inertia += np.sum(distances ** 2)

        return inertia

    def get_predicted_labels(self):
        """获取预测标签"""
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)

        regions = sorted(set(final_clusters_hashable.values()))
        region_to_label = {region: i for i, region in enumerate(regions)}

        labels = []
        for point in self.actual_points:
            if tuple(point) in final_clusters_hashable:
                labels.append(region_to_label[final_clusters_hashable[tuple(point)]])
            else:
                labels.append(-1)

        return labels

    def silhouette_score(self):
        """计算轮廓系数"""
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)
        
        coordinates = [point[:2] for point in final_clusters_hashable.keys()]
        regions = sorted(set(final_clusters_hashable.values()))
        region_to_label = {region: i for i, region in enumerate(regions)}
        labels = [region_to_label[final_clusters_hashable[point]] for point in final_clusters_hashable.keys()]

        X = np.array(coordinates)
        labels_np = np.array(labels)
        
        if len(regions) > 1:
            score = silhouette_score(X, labels_np, metric='euclidean')
        else:
            score = 0

        inertia = self.calculate_inertia(X, labels_np)

        print(f"等值线聚类轮廓系数: {score:.4f}")
        print(f"等值线聚类Inertia: {inertia:.2f}")

        return score, inertia

    def perform_clustering(self):
        """执行完整聚类流程"""
        self.generate_contour_areas()
        self.get_density_centers()
        self.assign_points_to_centers()
        return self.point_to_density_center

    def get_cluster_report(self):
        """获取聚类结果报告"""
        report = {}
        for center in self.density_centers:
            cluster_points = [
                point for point, centers in self.point_to_density_center.items()
                if center in centers
            ]
            total_load = sum(point[2] for point in cluster_points)
            report[center.center] = {
                'points': cluster_points,
                'total_load': total_load,
            }
        return report

    def get_final_cluster(self):
        """获取最终聚类结果"""
        return self.point_to_density_center


# 示例用法
if __name__ == "__main__":
    # 实际点数据 (x, y, 密度值, 客户ID)
    actual_points = [
        (0.2, 0.3, 1.5, 1),
        (0.8, 1.2, 2.0, 2),
        (1.5, 1.5, 3.0, 3),
        (2.3, 2.6, 1.0, 4),
        (3.3, 1.6, 1.8, 5),
        (8.3, 2.0, 2.5, 6),
        (9.3, 3.0, 2.5, 7),
        (5.3, 2.0, 1.3, 8),
        (3.3, 7.2, 1.4, 9)
    ]
    
    print("="*80)
    print("测试不同插值方法")
    print("="*80)
    
    # 测试不同插值方法
    methods = [
        ('linear', {}),
        ('cubic', {'cubic_smoothing': 0.5}),
        ('idw', {'idw_power': 2.0, 'idw_neighbors': 8})
    ]
    
    for method, params in methods:
        print(f"\n{'='*80}")
        print(f"测试插值方法: {method}")
        print(f"{'='*80}")
        
        # 创建聚类实例
        cluster = ContourBasedClusteringWithoutZ(
            actual_points,
            grid_size=1.0,
            max_buffer_distance=5,
            interpolation_method=method,
            **params
        )
        
        # 执行聚类
        clusters = cluster.perform_clustering()
        
        # 可视化结果
        cluster.visualize_clusters()
        
        # 计算评估指标
        cluster.silhouette_score()
        
        print(f"\n发现 {len(cluster.density_centers)} 个聚类中心")
