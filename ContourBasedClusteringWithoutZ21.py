import math
import numpy as np
from matplotlib.contour import ContourSet
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
import matplotlib
from shapely.geometry import Point, LineString
matplotlib.use('TkAgg')  # or 'Qt5Agg' (requires PyQt5)
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree
plt.rcParams['font.sans-serif'] = ['SimHei']   # SimHei
plt.rcParams['axes.unicode_minus'] = False     # Fix minus-sign rendering

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
        self.manual_points = []  # Stores the interactively added points
        self.contour_areas = []
        self.density_centers = []
        self.point_to_density_center = {}

        # Internal class definition
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
                # If only (x, y) is given, add z = 0.0 and id = -1
                self.manual_virtual_points.append((point[0], point[1], 0.0, -1))
            elif len(point) == 3:
                # If (x, y, z) is given, add id = -1
                self.manual_virtual_points.append((point[0], point[1], point[2], -1))
            elif len(point) == 4:
                # If the full (x, y, z, id) is given
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

        # Expand the extent so that all regions are covered
        i_start = min_x - 2 * self.grid_size
        i_end = max_x + 2 * self.grid_size
        j_start = min_y - 2 * self.grid_size
        j_end = max_y + 2 * self.grid_size

        virtual_points = []
        i = i_start
        while i < i_end:
            j = j_start
            while j < j_end:
                # Check whether the current cell contains a real point
                has_point = any(
                    i <= x < i + self.grid_size and
                    j <= y < j + self.grid_size
                    for (x, y, _, _) in self.actual_points
                )

                # Generate virtual points when there are no real points
                if not has_point:
                    virtual_x = i + self.grid_size / 2
                    virtual_y = j + self.grid_size / 2
                    virtual_points.append((virtual_x, virtual_y, 0.0, -1))  # -1 marks an automatically generated virtual point

                j += self.grid_size
            i += self.grid_size

        # Merge the predefined manual virtual points
        if self.manual_virtual_points:
            for point in self.manual_virtual_points:
                if len(point) == 2:
                    virtual_points.append((point[0], point[1], 0.0, -1))
                elif len(point) == 3:
                    virtual_points.append((point[0], point[1], point[2], -1))
                elif len(point) == 4:
                    virtual_points.append(point)
            print(f"已添加 {len(self.manual_virtual_points)} 个预设虚拟点")

        # ====== Interactive drawing interface ======
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        grid_count = (len(x_ticks) - 1) * (len(y_ticks) - 1)
        auto_virtual_count = len(virtual_points)
        print(f"网格总数: {grid_count}, 自动虚拟点数: {auto_virtual_count}")

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the grid lines
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        # Plot real points
        ax.scatter(x_coords, y_coords, c='red', s=50, label="实际点", zorder=3)
        
        # Plot the automatically generated virtual points
        if virtual_points:
            auto_vx = [p[0] for p in virtual_points]
            auto_vy = [p[1] for p in virtual_points]
            ax.scatter(auto_vx, auto_vy, c='blue', s=30, marker='x', label="自动虚拟点", zorder=2)

        # Set the title and caption
        ax.set_title(
            "网格显示\n请用【鼠标左键】在空白处点击添加零值点\n完成后【关闭窗口】继续",
            color='blue',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')

        # === Click-event handler ===
        def on_click(event):
            # Accept only left clicks inside the axes
            if event.inaxes == ax and event.button == 1:
                # Get the click coordinates
                new_x, new_y = event.xdata, event.ydata

                # Create a new zero point (-2 marks a virtual point added interactively)
                new_point = (new_x, new_y, 0.0, -2)

                # Add to the class attribute
                self.manual_points.append(new_point)

                # Draw the new point immediately
                ax.scatter(new_x, new_y, c='green', s=60, marker='+', 
                          label="_nolegend_", zorder=4)
                ax.text(new_x, new_y, " 0", fontsize=8, color='green')

                # Redraw the canvas
                fig.canvas.draw()
                print(f"已手动添加零值点: ({new_x:.2f}, {new_y:.2f})")

        # Connect the event
        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        # Block until the window is closed
        plt.show()

        # Merge all virtual points (automatic and interactive)
        self.virtual_points = virtual_points + self.manual_points

        # Return all merged virtual points and the grid ticks
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

        # Expand the extent
        i_start = min_x - self.grid_size
        i_end = max_x + self.grid_size
        j_start = min_y - self.grid_size
        j_end = max_y + self.grid_size

        # Build the grid
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        print(f"xgrid: {x_ticks}")
        print(f"ygrid: {y_ticks}")

        # Compute the number of grid cells
        grid_count = (len(x_ticks) - 1) * (len(y_ticks) - 1)
        print(f"网格总数: {grid_count}")

        # Plot the grid
        fig, ax = plt.subplots(figsize=(8, 6))
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        # Plot real points
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
        # Generate virtual points
        virtual_points, x_ticks, y_ticks = self.generate_virtual_points()  # Generate the virtual points and the grid coordinates X, Y

        all_points = self.actual_points + self.virtual_points

        # Prepare interpolation data
        x1 = np.array([p[0] for p in all_points])
        y1 = np.array([p[1] for p in all_points])
        z1 = np.array([p[2] for p in all_points])

        X, Y = np.meshgrid(x_ticks, y_ticks)

        # Grid interpolation
        zi = griddata((x1, y1), z1, (X, Y), method='linear')
        zi = np.nan_to_num(zi, nan=0)

        # Generate contours
        fig, ax = plt.subplots()
        contour_set = ax.contour(X, Y, zi, levels=10, colors='black')
        
        # Show the contour plot
        ax.set_title('Contour Map with Actual and Virtual Points')

        # Plot real points
        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        ax.scatter(x_coords, y_coords, c='red', s=50, label="Actual Points", zorder=3)

        # Plot virtual points
        if self.virtual_points:
            # Tell automatic virtual points from interactive ones
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

        # Convert to polygonal regions
        face_polygons = self.line_to_polygon(contour_set)

        # Create contour-region objects
        contour_areas = []
        for polygon in face_polygons:
            points_in_poly = []
            for point in self.actual_points:
                if self.is_point_in_region(point[0], point[1], polygon):
                    points_in_poly.append(point)
            if points_in_poly:  # Keep only regions that contain points
                contour_areas.append(self.ContourArea(polygon, points_in_poly))

        self.contour_areas = contour_areas

        # Build the region hierarchy
        for area in self.contour_areas:
            area_poly = Polygon(area.polygon)
            for other in self.contour_areas:
                if area == other:
                    continue
                other_poly = Polygon(other.polygon)
                if area_poly.within(other_poly):  # If area lies inside other, keep area.isLeave = false
                    area.parent = other
                    other.isLeave = False
                    area.childLength += 1

        return self.contour_areas

    def get_density_centers(self):
        """获取密度中心点（关键聚类中心）"""
        # Take the outermost ring; if it is the outermost, the region is a density centre
        outCircle = [area for area in self.contour_areas if area.parent is None]  # Take the largest ring
        self.density_centers = outCircle
        return outCircle

    def visualize_current_area(self, area, index, total):
        """可视化当前聚类区域及其特征"""
        plt.subplot((total + 1) // 2, 2, index + 1)  # Build the subplot layout dynamically

        # 1. Draw the polygon boundaries
        poly_patch = patches.Polygon(
            area.polygon,
            closed=True,
            fill=None,
            edgecolor='blue',
            linewidth=1.5,
            linestyle='--'
        )
        plt.gca().add_patch(poly_patch)

        # 2. Mark the real points inside the region
        points_x = [p[0] for p in area.points]
        points_y = [p[1] for p in area.points]
        plt.scatter(points_x, points_y, c='red', s=60,
                    label=f'Points: {len(area.points)}')

        # 3. Mark the region centres
        plt.scatter(area.center[0], area.center[1],
                    c='green', marker='*', s=200,
                    label='Centroid')

        # 4. Add information labels
        plt.annotate(f"Cargo Vol: {area.cargoVol}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow"))

        # 5. Set figure properties
        plt.title(f"Leaf Area #{index + 1} at ({area.center[0]:.1f}, {area.center[1]:.1f})")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')

        # 6. Adapt the axis limits
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

        # Grow the buffer step by step
        for buffer_dist in range(0, self.max_buffer_distance + 1):
            current_assignments = {}

            # Create a buffer for each density centre
            for center in self.density_centers:
                center_poly = Polygon(center.polygon)
                buffer_zone = center_poly.buffer(buffer_dist)

                # Find the points inside the buffer
                for point in self.actual_points:
                    if point in assigned_points:
                        continue

                    pt = Point(point[0], point[1])
                    if buffer_zone.contains(pt):
                        if point not in current_assignments:
                            current_assignments[point] = []
                        current_assignments[point].append(center)

            # Resolve overlapping assignments (a point inside several buffers)
            for point, centers in current_assignments.items():
                if len(centers) > 1:
                    # Choose the nearest centre
                    min_dist = float('inf')
                    closest_center = None
                    for center in centers:
                        dist = self.calculate_distance(point, center.center)
                        if dist < min_dist:
                            min_dist = dist
                            closest_center = center
                    current_assignments[point] = [closest_center]

                # Record the final assignment
                self.point_to_density_center[point] = current_assignments[point]
                assigned_points.add(point)

        return self.point_to_density_center

    def visualize_clusters(self):
        """可视化聚类结果"""
        if not self.density_centers:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

        # Plot the density centres and buffers
        for i, center in enumerate(self.density_centers):
            color = colors[i % len(colors)]

            # Plot the original regions
            poly_patch = patches.Polygon(
                center.polygon,
                closed=True,
                fill=None,
                edgecolor=color,
                linestyle='dashed'
            )
            ax.add_patch(poly_patch)

            # Plot the real points assigned to this centre
            for point, centers in self.point_to_density_center.items():
                if center in centers:
                    ax.plot(point[0], point[1], 'o', color=color, markersize=6)

        # Plot unassigned points
        all_assigned = {point for point in self.point_to_density_center}
        for point in self.actual_points:
            if point not in all_assigned:
                ax.plot(point[0], point[1], 'ko', markersize=6)  # Black marks unassigned points

        # Set figure properties
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
        n_clusters = len(np.unique(labels))  # Get the number of clusters
        inertia = 0.0  # Initialise inertia

        for k in range(n_clusters):
            # Collect the points of cluster k
            cluster_points = X[labels == k]

            # Compute the centre (mean) of cluster k
            cluster_center = np.mean(cluster_points, axis=0)

            # Sum of squared distances from all cluster points to the cluster centre
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)  # Euclidean distance
            inertia += np.sum(distances ** 2)  # Accumulate the squared distance of each point

        return inertia

    def get_predicted_labels(self):
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            # If the region list holds hashable elements, convert it to a tuple
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                # If a region is a custom object, try its object id as the key
                # Otherwise build a unique key from the object attributes
                # Simplified here by using the object id
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)

        # Get the unique region labels
        regions = sorted(set(final_clusters_hashable.values()))
        region_to_label = {region: i for i, region in enumerate(regions)}

        # Process all points and label noise as -1
        labels = []
        for point in self.actual_points:
            # If the point is in final_clusters_hashable, assign a valid label
            if tuple(point) in final_clusters_hashable:
                labels.append(region_to_label[final_clusters_hashable[tuple(point)]])
            else:
                labels.append(-1)  # A point in no cluster is labelled -1 as noise

        return labels

    # Compute the silhouette coefficient
    def silhouette_score(self):
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            # If the region list holds hashable elements, convert it to a tuple
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                # If a region is a custom object, try its object id as the key
                # Otherwise build a unique key from the object attributes
                # Simplified here by using the object id
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)
        coordinates = [point[:2] for point in final_clusters_hashable.keys()]

        # Step 2: get all unique region labels
        regions = sorted(set(final_clusters_hashable.values()))

        # Map regions to numeric labels
        region_to_label = {region: i for i, region in enumerate(regions)}

        # Step 3: build the label array in the same order as the coordinates
        labels = [region_to_label[final_clusters_hashable[point]] for point in final_clusters_hashable.keys()]

        # Convert to a NumPy array
        X = np.array(coordinates)  # Shape: (n_points, 2)
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


# Example usage
if __name__ == "__main__":
    # Real point data (x, y, cargo volume, customer id)
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
    
    # Add manual virtual points (optional)
    manual_virtual_points = [
        (5.0, 5.0),  # Only (x, y) is given
        (6.0, 6.0, 0.5),  # (x, y, z) is given
        (7.0, 7.0, 0.3, -2),  # The full (x, y, z, id) is given
    ]
    
    # Create clustering instance
    cluster = ContourBasedClusteringWithoutZ(
        actual_points, 
        grid_size=1.0,
        max_buffer_distance=5,
        manual_virtual_points=manual_virtual_points
    )
    
    # More virtual points can also be added after creation
    cluster.add_manual_virtual_points([
        (4.0, 4.0),
        (10.0, 10.0, 0.2)
    ])
    
    # Run clustering
    clusters = cluster.perform_clustering()

    # Visualize results
    cluster.visualize_clusters()

    # Get the clustering report
    report = cluster.get_cluster_report()
    for center, data in report.items():
        print(f"\nCluster Center at {center}:")
        print(f"Total Load: {data['total_load']:.2f}")
        print("Points:")
        for point in data['points']:
            print(f"  Client {point[3]} at ({point[0]}, {point[1]}) with load {point[2]}")
