import math
import numpy as np
from matplotlib.contour import ContourSet
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
import matplotlib
from shapely.geometry import Point, LineString
matplotlib.use('TkAgg')  # 或 'Qt5Agg'（需安装PyQt5）
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree

plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
class ContourBasedClusteringWithoutZ:
    def __init__(self, actual_points, grid_size=1.0, max_buffer_distance=5, 
                 manual_virtual_points=None, min_cluster_size=5):
        """
        初始化聚类算法

        参数:
        actual_points -- 实际点列表，格式为[(x, y, z, clientId), ...]
        grid_size -- 生成虚拟点的网格大小 (默认1.0) 值越小，网格越密集，导致虚拟点增加，计算量变大
        max_buffer_distance -- 最大缓冲区距离 (默认5)
        manual_virtual_points -- 手动添加的虚拟点列表，格式为[(x, y), ...] 或 [(x, y, z), ...]
                                如果提供，这些点将被添加到自动生成的虚拟点中
        min_cluster_size -- 最小簇大小阈值，小于此值的簇将被合并到最近的簇中 (默认5)
        """
        self.actual_points = actual_points
        self.grid_size = grid_size
        self.max_buffer_distance = int(max_buffer_distance)
        self.virtual_points = []
        self.manual_virtual_points = manual_virtual_points or []
        self.manual_points = []  # 用于存储交互式添加的点
        self.contour_areas = []
        self.density_centers = []
        self.point_to_density_center = {}
        self.min_cluster_size = min_cluster_size  # 新增：最小簇大小阈值

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

    def add_manual_virtual_points(self, points):
        """
        手动添加虚拟点
        
        参数:
        points -- 虚拟点列表，格式为[(x, y), ...] 或 [(x, y, z), ...]
        """
        if not points:
            return
        
        for point in points:
            if len(point) == 2:
                # 如果只提供 (x, y)，自动添加 z=0.0 和 id=-1
                self.manual_virtual_points.append((point[0], point[1], 0.0, -1))
            elif len(point) == 3:
                # 如果提供 (x, y, z)，自动添加 id=-1
                self.manual_virtual_points.append((point[0], point[1], point[2], -1))
            elif len(point) == 4:
                # 如果提供完整的 (x, y, z, id)
                self.manual_virtual_points.append(point)
            else:
                print(f"警告: 忽略格式不正确的点 {point}")

    def generate_virtual_points(self):
        """生成虚拟坐标点以完善空间分布，并支持交互式添加虚拟点"""
        if not self.actual_points:
            return []

        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 扩展边界确保覆盖所有区域
        i_start = min_x - 2 * self.grid_size
        i_end = max_x + 2 * self.grid_size
        j_start = min_y - 2 * self.grid_size
        j_end = max_y + 2 * self.grid_size

        virtual_points = []
        i = i_start
        while i < i_end:
            j = j_start
            while j < j_end:
                # 检查当前网格是否存在实际点
                has_point = any(
                    i <= x < i + self.grid_size and
                    j <= y < j + self.grid_size
                    for (x, y, _, _) in self.actual_points
                )

                # 无实际点时生成虚拟点
                if not has_point:
                    virtual_x = i + self.grid_size / 2
                    virtual_y = j + self.grid_size / 2
                    virtual_points.append((virtual_x, virtual_y, 0.0, -1))  # -1表示自动虚拟点

                j += self.grid_size
            i += self.grid_size

        # 合并预设的手动虚拟点
        if self.manual_virtual_points:
            for point in self.manual_virtual_points:
                if len(point) == 2:
                    virtual_points.append((point[0], point[1], 0.0, -1))
                elif len(point) == 3:
                    virtual_points.append((point[0], point[1], point[2], -1))
                elif len(point) == 4:
                    virtual_points.append(point)
            print(f"已添加 {len(self.manual_virtual_points)} 个预设虚拟点")

        # ====== 绘制交互界面 ======
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        grid_count = (len(x_ticks) - 1) * (len(y_ticks) - 1)
        auto_virtual_count = len(virtual_points)
        print(f"网格总数: {grid_count}, 自动虚拟点数: {auto_virtual_count}")

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制网格线
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        # 绘制实际点
        ax.scatter(x_coords, y_coords, c='red', s=50, label="实际点", zorder=3)
        ax.set_aspect('equal')
        # 绘制自动生成的虚拟点
        if virtual_points:
            auto_vx = [p[0] for p in virtual_points]
            auto_vy = [p[1] for p in virtual_points]
            ax.scatter(auto_vx, auto_vy, c='blue', s=30, marker='x', label="自动虚拟点", zorder=2)

        # 设置标题和说明
        ax.set_title(
            "网格显示\n请用【鼠标左键】在空白处点击添加零值点\n完成后【关闭窗口】继续",
            color='blue',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')

        # === 定义点击事件处理函数 ===
        def on_click(event):
            # 确保点击在坐标轴内，且是左键
            if event.inaxes == ax and event.button == 1:
                # 获取点击坐标
                new_x, new_y = event.xdata, event.ydata

                # 创建新的零值点 (-2 表示交互式手动添加的虚拟点)
                new_point = (new_x, new_y, 0.0, -2)

                # 添加到类属性中
                self.manual_points.append(new_point)

                # 实时绘制新点
                ax.scatter(new_x, new_y, c='green', s=60, marker='+', 
                          label="_nolegend_", zorder=4)
                ax.text(new_x, new_y, " 0", fontsize=8, color='green')

                # 刷新画布
                fig.canvas.draw()
                print(f"已手动添加零值点: ({new_x:.2f}, {new_y:.2f})")

        # 连接事件
        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        # 阻塞显示，直到用户关闭窗口
        plt.show()

        # 合并所有虚拟点（自动 + 交互式手动）
        self.virtual_points = virtual_points + self.manual_points

        # 返回合并后的所有虚拟点以及网格刻度
        print(f"交互结束。自动点: {auto_virtual_count}, 交互式手动点: {len(self.manual_points)}, 总虚拟点数: {len(self.virtual_points)}")

        return self.virtual_points, x_ticks, y_ticks

    def visualize_grid(self):
        """绘制网格并显示网格数量"""
        if not self.actual_points:
            print("没有实际点，无法绘制网格")
            return

        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 扩展边界
        i_start = min_x - self.grid_size
        i_end = max_x + self.grid_size
        j_start = min_y - self.grid_size
        j_end = max_y + self.grid_size

        # 生成网格
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        print(f"xgrid: {x_ticks}")
        print(f"ygrid: {y_ticks}")

        # 计算网格数量
        grid_count = (len(x_ticks) - 1) * (len(y_ticks) - 1)
        print(f"网格总数: {grid_count}")

        # 绘制网格
        fig, ax = plt.subplots(figsize=(8, 6))
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        # 绘制实际点
        ax.scatter(x_coords, y_coords, c='red', s=50, label="Actual Points")
        ax.set_title(f"网格显示 (总数: {grid_count})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

        return grid_count

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
        """生成等值线区域并建立层次结构"""
        # 生成虚拟点
        virtual_points, x_ticks, y_ticks = self.generate_virtual_points()

        all_points = self.actual_points + self.virtual_points

        # 准备插值数据
        x1 = np.array([p[0] for p in all_points])
        y1 = np.array([p[1] for p in all_points])
        z1 = np.array([p[2] for p in all_points])

        X, Y = np.meshgrid(x_ticks, y_ticks)

        # 网格插值
        zi = griddata((x1, y1), z1, (X, Y), method='linear')
        zi = np.nan_to_num(zi, nan=0)

        # 生成等值线
        fig, ax = plt.subplots()
        contour_set = ax.contour(X, Y, zi, levels=10, colors='black')
        
        # 展示等值线图
        ax.set_title('Contour Map with Actual and Virtual Points')

        # 绘制实际点
        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        ax.scatter(x_coords, y_coords, c='red', s=50, label="Actual Points", zorder=3)

        # 绘制虚拟点
        if self.virtual_points:
            # 区分自动虚拟点和交互式手动点
            auto_points = [p for p in self.virtual_points if p[3] == -1]
            manual_points = [p for p in self.virtual_points if p[3] == -2]
            
            if auto_points:
                auto_vx = [p[0] for p in auto_points]
                auto_vy = [p[1] for p in auto_points]
                ax.scatter(auto_vx, auto_vy, c='blue', s=30, marker='x', 
                          label="Auto Virtual Points", zorder=2)
            
            if manual_points:
                manual_vx = [p[0] for p in manual_points]
                manual_vy = [p[1] for p in manual_points]
                ax.scatter(manual_vx, manual_vy, c='green', s=50, marker='+', 
                          label="Manual Virtual Points", zorder=2)
        
        ax.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
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
            if points_in_poly:  # 仅包含有点的区域
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
        # 获取到最外层的圈(最外圈则为密度中心)
        outCircle = [area for area in self.contour_areas if area.parent is None]
        self.density_centers = outCircle
        return outCircle

    def assign_points_to_centers(self):
        """通过缓冲区扩展将点分配给密度中心"""
        if not self.density_centers:
            return {}

        point_assignments = {}
        assigned_points = set()

        # 逐步扩展缓冲区
        for buffer_dist in range(0, self.max_buffer_distance + 1):
            current_assignments = {}

            # 为每个密度中心创建缓冲区
            for center in self.density_centers:
                center_poly = Polygon(center.polygon)
                buffer_zone = center_poly.buffer(buffer_dist)

                # 查找缓冲区内的点
                for point in self.actual_points:
                    if point in assigned_points:
                        continue

                    pt = Point(point[0], point[1])
                    if buffer_zone.contains(pt):
                        if point not in current_assignments:
                            current_assignments[point] = []
                        current_assignments[point].append(center)

            # 处理重叠分配（多个中心都包含同一个点）
            for point, centers in current_assignments.items():
                if len(centers) > 1:
                    # 选择距离最近的中心
                    min_dist = float('inf')
                    closest_center = None
                    for center in centers:
                        dist = self.calculate_distance(point, center.center)
                        if dist < min_dist:
                            min_dist = dist
                            closest_center = center
                    current_assignments[point] = [closest_center]

                # 记录最终分配
                self.point_to_density_center[point] = current_assignments[point]
                assigned_points.add(point)

        return self.point_to_density_center

    def assign_unassigned_points(self):
        """
        将无归属的点分配到最近的簇中
        对于没有被分配到任何簇的点，找到距离它最近的簇中的点，然后将该点分配给该簇
        
        返回:
        assigned_count: 被分配的无归属点数量
        """
        if not self.density_centers:
            print("警告: 没有密度中心，无法分配无归属点")
            return 0
        
        # 找出所有已分配的点
        assigned_points = set(self.point_to_density_center.keys())
        
        # 找出所有未分配的点
        unassigned_points = [point for point in self.actual_points 
                            if point not in assigned_points]
        
        if not unassigned_points:
            print("所有点都已分配，无需处理无归属点")
            return 0
        
        print(f"\n发现 {len(unassigned_points)} 个无归属点")
        
        # 获取所有簇及其包含的点
        cluster_points_map = {}
        for center in self.density_centers:
            cluster_points = [point for point, centers in self.point_to_density_center.items() 
                             if center in centers]
            if cluster_points:  # 只考虑非空簇
                cluster_points_map[center] = cluster_points
        
        if not cluster_points_map:
            print("警告: 没有非空簇可以分配无归属点")
            return 0
        
        assigned_count = 0
        
        # 对每个无归属点，找到最近的簇
        for unassigned_point in unassigned_points:
            min_dist = float('inf')
            nearest_center = None
            
            # 遍历所有簇，找到距离该点最近的簇中的点
            for center, cluster_points in cluster_points_map.items():
                # 计算该点到簇中所有点的最小距离
                for cluster_point in cluster_points:
                    dist = self.calculate_distance(unassigned_point, cluster_point)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_center = center
            
            if nearest_center is None:
                print(f"  警告: 点 {unassigned_point[:2]} 找不到合适的簇，保持未分配")
                continue
            
            # 将该点分配给最近的簇
            self.point_to_density_center[unassigned_point] = [nearest_center]
            assigned_count += 1
        
        print(f"已将 {assigned_count} 个无归属点分配到最近的簇中")
        return assigned_count

    def merge_small_clusters(self):
        """
        合并小簇：将点数小于min_cluster_size的簇合并到最近的大簇中
        注意：只能合并到点数 >= min_cluster_size 的簇中
        合并规则：小簇中的每个点，找到距离它最近的大簇中的点，然后将该点分配给该大簇
        
        返回:
        merged_count: 被合并的小簇数量
        """
        if not self.point_to_density_center or not self.density_centers:
            print("警告: 没有聚类结果，无法合并小簇")
            return 0
        
        # 统计每个密度中心的点数（初始统计）
        center_point_counts = {}
        for center in self.density_centers:
            count = sum(1 for point, centers in self.point_to_density_center.items() 
                       if center in centers)
            center_point_counts[center] = count
        
        # 找出小簇（点数 < min_cluster_size）
        small_clusters = [center for center, count in center_point_counts.items() 
                         if count < self.min_cluster_size and count > 0]
        
        if not small_clusters:
            print(f"没有小于 {self.min_cluster_size} 个点的簇，无需合并")
            return 0
        
        print(f"\n发现 {len(small_clusters)} 个小簇（点数 < {self.min_cluster_size}）")
        
        # 找出大簇（点数 >= min_cluster_size）及其包含的点
        large_clusters = {}
        for center, count in center_point_counts.items():
            if count >= self.min_cluster_size:
                # 获取该大簇中的所有点
                cluster_points = [point for point, centers in self.point_to_density_center.items() 
                                 if center in centers]
                large_clusters[center] = cluster_points
        
        if not large_clusters:
            print(f"警告: 没有点数 >= {self.min_cluster_size} 的簇可以合并，保持原样")
            return 0
        
        print(f"找到 {len(large_clusters)} 个大簇（点数 >= {self.min_cluster_size}）可用于合并")
        
        merged_count = 0
        total_reassigned_points = 0
        
        # 对每个小簇，将其中的每个点分配给最近的大簇
        for small_center in small_clusters:
            # 获取小簇中的所有点
            small_cluster_points = [point for point, centers in self.point_to_density_center.items() 
                                   if small_center in centers]
            
            if not small_cluster_points:
                continue
            
            print(f"\n  处理小簇 (中心: {small_center.center}, 点数: {len(small_cluster_points)})")
            
            # 对小簇中的每个点，找到最近的大簇
            reassigned_points = 0
            for small_point in small_cluster_points:
                min_dist = float('inf')
                nearest_large_center = None
                
                # 遍历所有大簇，找到距离该点最近的大簇中的点
                for large_center, large_points in large_clusters.items():
                    # 计算该点到大簇中所有点的最小距离
                    for large_point in large_points:
                        dist = self.calculate_distance(small_point, large_point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_large_center = large_center
                
                if nearest_large_center is None:
                    print(f"    警告: 点 {small_point[:2]} 找不到合适的大簇，保持不变")
                    continue
                
                # 将该点重新分配给最近的大簇
                self.point_to_density_center[small_point] = [nearest_large_center]
                reassigned_points += 1
            
            if reassigned_points > 0:
                print(f"    已将 {reassigned_points} 个点重新分配到最近的大簇")
                merged_count += 1
                total_reassigned_points += reassigned_points
        
        print(f"\n共处理了 {merged_count} 个小簇，重新分配了 {total_reassigned_points} 个点")
        return merged_count

    def visualize_clusters(self):
        """可视化聚类结果"""
        if not self.density_centers:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 
                 'brown', 'pink', 'gray', 'olive', 'navy']

        # 统计每个中心的点数
        center_point_counts = {}
        for center in self.density_centers:
            count = sum(1 for point, centers in self.point_to_density_center.items() 
                       if center in centers)
            center_point_counts[center] = count

        # 绘制密度中心和缓冲区
        for i, center in enumerate(self.density_centers):
            point_count = center_point_counts.get(center, 0)
            if point_count == 0:
                continue  # 跳过没有点的簇
            
            color = colors[i % len(colors)]

            # 绘制原始区域
            poly_patch = patches.Polygon(
                center.polygon,
                closed=True,
                fill=None,
                edgecolor=color,
                linestyle='dashed'
            )
            ax.add_patch(poly_patch)

            # 绘制属于该中心的实际点
            for point, centers in self.point_to_density_center.items():
                if center in centers:
                    ax.plot(point[0], point[1], 'o', color=color, markersize=6)
            
            # 标注簇的点数
            ax.text(center.center[0], center.center[1], f'n={point_count}', 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 绘制未分配的点
        all_assigned = {point for point in self.point_to_density_center}
        unassigned_count = 0
        for point in self.actual_points:
            if point not in all_assigned:
                ax.plot(point[0], point[1], 'ko', markersize=6)
                unassigned_count += 1

        # 设置图形属性
        title = 'Contour-Based Clustering Result'
        if self.min_cluster_size > 1:
            title += f' (min_cluster_size={self.min_cluster_size})'
        if unassigned_count > 0:
            title += f'\n未分配点: {unassigned_count}'
        
        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def calculate_inertia(self, X, labels):
        """计算簇内误差平方和（Inertia）"""
        n_clusters = len(np.unique(labels))
        inertia = 0.0

        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                continue
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

        # 获取唯一的区域标签
        regions = sorted(set(final_clusters_hashable.values()))
        region_to_label = {region: i for i, region in enumerate(regions)}

        # 处理所有点，给噪声点分配 -1 标签
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

        print("等值线聚类轮廓系数:", score)
        print("等值线聚类方差:", inertia)

        return score, inertia

    def perform_clustering(self):
        """执行完整聚类流程（包含小簇合并和无归属点分配）"""
        self.generate_contour_areas()
        self.get_density_centers()
        self.assign_points_to_centers()
        
        # 新增：合并小簇
        if self.min_cluster_size > 1:
            self.merge_small_clusters()
        
        # 新增：分配无归属点到最近的簇
        self.assign_unassigned_points()
        
        return self.point_to_density_center

    def get_cluster_report(self):
        """获取聚类结果报告"""
        report = {}
        for center in self.density_centers:
            cluster_points = [
                point for point, centers in self.point_to_density_center.items()
                if center in centers
            ]
            if not cluster_points:  # 跳过空簇
                continue
            total_load = sum(point[2] for point in cluster_points)
            report[center.center] = {
                'points': cluster_points,
                'total_load': total_load,
                'point_count': len(cluster_points)
            }
        return report

    def get_final_cluster(self):
        return self.point_to_density_center



# 示例用法
if __name__ == "__main__":
    # 实际点数据 (x, y, 货物量, 客户ID)
    actual_points = [
        (0.2, 0.3, 1.5, 1),
        (0.8, 1.2, 2.0, 2),
        (1.5, 1.5, 3.0, 3),
        (2.3, 2.6, 1.0, 4),
        (3.3, 1.6, 1.8, 5),
        (8.3, 2.0, 2.5, 6),
        (9.3, 3.0, 2.5, 7),
        (5.3, 2.0, 1.3, 8),
        (3.3, 7.2, 1.4, 9),
        # 添加一些孤立点（形成小簇）
        (15.0, 15.0, 1.0, 10),
        (15.5, 15.2, 1.2, 11),
        (16.0, 15.5, 0.8, 12),
    ]
    
    # 手动添加虚拟点（可选）
    manual_virtual_points = [
        (5.0, 5.0),
        (6.0, 6.0, 0.5),
        (7.0, 7.0, 0.3, -2),
    ]
    
    # 创建聚类实例（设置最小簇大小为5）
    cluster = ContourBasedClusteringWithoutZ(
        actual_points, 
        grid_size=1.0,
        max_buffer_distance=5,
        manual_virtual_points=manual_virtual_points,
        min_cluster_size=5  # 小于5个点的簇将被合并
    )
    
    # 执行聚类（会自动合并小簇）
    clusters = cluster.perform_clustering()

    # 可视化结果
    cluster.visualize_clusters()

    # 获取聚类报告
    report = cluster.get_cluster_report()
    print("\n" + "="*60)
    print("聚类报告")
    print("="*60)
    for center, data in report.items():
        print(f"\n簇中心位于 {center}:")
        print(f"  点数: {data['point_count']}")
        print(f"  总货物量: {data['total_load']:.2f}")
        print("  包含的点:")
        for point in data['points']:
            print(f"    客户 {point[3]} 位于 ({point[0]}, {point[1]}) 货物量 {point[2]}")
    print("="*60)
