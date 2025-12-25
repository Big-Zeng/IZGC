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


class ContourBasedClusteringWithoutZ:
    def __init__(self, actual_points, grid_size=1.0, max_buffer_distance=5, manual_virtual_points=None):
        """
        初始化聚类算法

        参数:
        actual_points -- 实际点列表，格式为[(x, y, z, clientId), ...]
        grid_size -- 生成虚拟点的网格大小 (默认1.0) 值越小，网格越密集，导致虚拟点增加，计算量变大
        max_buffer_distance -- 最大缓冲区距离 (默认5)
        manual_virtual_points -- 手动添加的虚拟点列表，格式为[(x, y), ...] 或 [(x, y, z), ...]
                                如果提供，这些点将被添加到自动生成的虚拟点中
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
        virtual_points, x_ticks, y_ticks = self.generate_virtual_points()  # 生成虚拟点、网格X,Y的坐标值

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
                if area_poly.within(other_poly):  # area 在 other内部的话，area.isLeave=false
                    area.parent = other
                    other.isLeave = False
                    area.childLength += 1

        return self.contour_areas

    def get_density_centers(self):
        """获取密度中心点（关键聚类中心）"""
        # 获取到最外层的圈(最外圈则为密度中心)
        outCircle = [area for area in self.contour_areas if area.parent is None]  # 获取到最大的圈
        self.density_centers = outCircle
        return outCircle

    def visualize_current_area(self, area, index, total):
        """可视化当前聚类区域及其特征"""
        plt.subplot((total + 1) // 2, 2, index + 1)  # 动态创建子图布局

        # 1. 绘制多边形边界
        poly_patch = patches.Polygon(
            area.polygon,
            closed=True,
            fill=None,
            edgecolor='blue',
            linewidth=1.5,
            linestyle='--'
        )
        plt.gca().add_patch(poly_patch)

        # 2. 标记区域内实际点
        points_x = [p[0] for p in area.points]
        points_y = [p[1] for p in area.points]
        plt.scatter(points_x, points_y, c='red', s=60,
                    label=f'Points: {len(area.points)}')

        # 3. 标记区域中心
        plt.scatter(area.center[0], area.center[1],
                    c='green', marker='*', s=200,
                    label='Centroid')

        # 4. 添加信息标注
        plt.annotate(f"Cargo Vol: {area.cargoVol}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow"))

        # 5. 设置图形属性
        plt.title(f"Leaf Area #{index + 1} at ({area.center[0]:.1f}, {area.center[1]:.1f})")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')

        # 6. 自适应坐标范围
        min_x = min(p[0] for p in area.polygon)
        max_x = max(p[0] for p in area.polygon)
        min_y = min(p[1] for p in area.polygon)
        max_y = max(p[1] for p in area.polygon)
        plt.xlim(min_x - 1, max_x + 1)
        plt.ylim(min_y - 1, max_y + 1)

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

    def visualize_clusters(self):
        """可视化聚类结果"""
        if not self.density_centers:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

        # 绘制密度中心和缓冲区
        for i, center in enumerate(self.density_centers):
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

        # 绘制未分配的点
        all_assigned = {point for point in self.point_to_density_center}
        for point in self.actual_points:
            if point not in all_assigned:
                ax.plot(point[0], point[1], 'ko', markersize=6)  # 黑色表示未分配

        # 设置图形属性
        ax.set_title('Contour-Based Clustering Result')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def calculate_inertia(self, X, labels):
        """
        计算簇内误差平方和（Inertia）

        参数:
        X : ndarray, shape (n_samples, n_features)
            输入的样本数据。
        labels : ndarray, shape (n_samples,)
            每个样本所属的簇标签。

        返回:
        inertia : float
            聚类结果的簇内误差平方和（Inertia）。
        """
        n_clusters = len(np.unique(labels))  # 获取簇的个数
        inertia = 0.0  # 初始化 Inertia

        for k in range(n_clusters):
            # 获取属于簇 k 的所有点
            cluster_points = X[labels == k]

            # 计算簇 k 的中心（均值）
            cluster_center = np.mean(cluster_points, axis=0)

            # 计算簇内所有点到簇中心的距离平方和
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)  # 欧几里得距离
            inertia += np.sum(distances ** 2)  # 累加每个点的距离平方

        return inertia

    def get_predicted_labels(self):
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            # 如果区域列表包含可哈希元素，将其转换为元组
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                # 如果区域是自定义对象，尝试使用对象ID作为标识
                # 或者根据对象属性创建唯一标识
                # 这里简化处理为使用对象ID
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)

        # 获取唯一的区域标签
        regions = sorted(set(final_clusters_hashable.values()))
        region_to_label = {region: i for i, region in enumerate(regions)}

        # 处理所有点，给噪声点分配 -1 标签
        labels = []
        for point in self.actual_points:
            # 如果该点在 final_clusters_hashable 中，分配一个有效标签
            if tuple(point) in final_clusters_hashable:
                labels.append(region_to_label[final_clusters_hashable[tuple(point)]])
            else:
                labels.append(-1)  # 如果点不在任何聚类中，标记为噪声点，标签为 -1

        return labels

    # 计算轮廓系数
    def silhouette_score(self):
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            # 如果区域列表包含可哈希元素，将其转换为元组
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                # 如果区域是自定义对象，尝试使用对象ID作为标识
                # 或者根据对象属性创建唯一标识
                # 这里简化处理为使用对象ID
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)
        coordinates = [point[:2] for point in final_clusters_hashable.keys()]

        # 步骤2: 获取所有唯一的区域标签
        regions = sorted(set(final_clusters_hashable.values()))

        # 创建区域到数字标签的映射
        region_to_label = {region: i for i, region in enumerate(regions)}

        # 步骤3: 创建标签数组（顺序与坐标数组一致）
        labels = [region_to_label[final_clusters_hashable[point]] for point in final_clusters_hashable.keys()]

        # 转换为NumPy数组
        X = np.array(coordinates)  # 形状：(n_points, 2)
        labels_np = np.array(labels)
        if len(regions) > 1:
            score = silhouette_score(X, labels_np, metric='euclidean')
        else:
            score = 0

        inertia = self.calculate_inertia(X, labels_np)

        print("等值线聚类轮廓系数1:", score)
        print("等值线聚类方差2:", inertia)

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
        (3.3, 7.2, 1.4, 9)
    ]
    
    # 手动添加虚拟点（可选）
    manual_virtual_points = [
        (5.0, 5.0),  # 只提供 (x, y)
        (6.0, 6.0, 0.5),  # 提供 (x, y, z)
        (7.0, 7.0, 0.3, -2),  # 提供完整的 (x, y, z, id)
    ]
    
    # 创建聚类实例
    cluster = ContourBasedClusteringWithoutZ(
        actual_points, 
        grid_size=1.0,
        max_buffer_distance=5,
        manual_virtual_points=manual_virtual_points
    )
    
    # 也可以在创建后添加更多虚拟点
    cluster.add_manual_virtual_points([
        (4.0, 4.0),
        (10.0, 10.0, 0.2)
    ])
    
    # 执行聚类
    clusters = cluster.perform_clustering()

    # 可视化结果
    cluster.visualize_clusters()

    # 获取聚类报告
    report = cluster.get_cluster_report()
    for center, data in report.items():
        print(f"\nCluster Center at {center}:")
        print(f"Total Load: {data['total_load']:.2f}")
        print("Points:")
        for point in data['points']:
            print(f"  Client {point[3]} at ({point[0]}, {point[1]}) with load {point[2]}")
