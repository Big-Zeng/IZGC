import math
import numpy as np
from matplotlib.contour import ContourSet
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
import matplotlib
from shapely.geometry import Point, LineString

# Switch to TkAgg only when no backend has been fixed yet
if matplotlib.get_backend() == 'agg':
    pass  # keep the Agg backend
else:
    try:
        matplotlib.use('TkAgg')  # or 'Qt5Agg' (requires PyQt5)
    except:
        pass
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree

plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei (CJK sans-serif)
plt.rcParams['axes.unicode_minus'] = False  # render the minus sign correctly


class ContourBasedClusteringWithoutZ:
    def __init__(self, actual_points, grid_size=1.0, max_buffer_distance=5,
                 manual_virtual_points=None, min_cluster_size=5, interactive=True,
                 merge_small_clusters=True):
        """
        Initialize the clustering algorithm.

        Parameters:
        actual_points -- observed samples as [(x, y, z, clientId), ...]
        grid_size -- grid step for virtual points (default 1.0); a smaller step gives a denser grid and a higher cost
        max_buffer_distance -- maximum buffer distance (default 5)
        manual_virtual_points -- user-supplied virtual points as [(x, y), ...] or [(x, y, z), ...]
                                appended to the automatically generated virtual points
        min_cluster_size -- threshold below which a cluster counts as small (default 5)
        interactive -- show the interactive window (default True); set False for batch runs
        merge_small_clusters -- how small clusters are handled (default True)
                               True: merge each small cluster into the nearest large one
                               False: label the points of small clusters as noise (-1)
        """
        self.actual_points = actual_points
        self.grid_size = grid_size
        self.max_buffer_distance = int(max_buffer_distance)
        self.virtual_points = []
        self.manual_virtual_points = manual_virtual_points or []
        self.manual_points = []  # points added interactively by the user
        self.contour_areas = []
        self.density_centers = []
        self.point_to_density_center = {}
        self.min_cluster_size = int(min_cluster_size)  # small-cluster threshold
        self.interactive = interactive  # interactive display flag
        self.merge_small_clusters_flag = merge_small_clusters  # small-cluster handling mode

        # nested class definition
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
                """Total load carried by the points inside the region."""
                return sum(point[2] for point in self.points)

            def get_centroid(self):
                """Centroid of the region."""
                x_sum = sum(point[0] for point in self.points)
                y_sum = sum(point[1] for point in self.points)
                return (round(x_sum / len(self.points), 2), round(y_sum / len(self.points), 2))

        self.ContourArea = ContourArea

    def add_manual_virtual_points(self, points):
        """
        Add virtual points manually.

        Parameters:
        points -- virtual points as [(x, y), ...] or [(x, y, z), ...]
        """
        if not points:
            return

        for point in points:
            if len(point) == 2:
                # (x, y) only: append z=0.0 and id=-1
                self.manual_virtual_points.append((point[0], point[1], 0.0, -1))
            elif len(point) == 3:
                # (x, y, z): append id=-1
                self.manual_virtual_points.append((point[0], point[1], point[2], -1))
            elif len(point) == 4:
                # complete (x, y, z, id) tuple
                self.manual_virtual_points.append(point)
            else:
                print(f"Warning: skipping malformed point {point}")

    def generate_virtual_points(self):
        """Generate virtual points over vacant cells and allow interactive insertion."""
        if not self.actual_points:
            return []

        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # pad the bounding box so the grid covers the whole extent
        i_start = min_x - 2 * self.grid_size
        i_end = max_x + 2 * self.grid_size
        j_start = min_y - 2 * self.grid_size
        j_end = max_y + 2 * self.grid_size

        virtual_points = []
        i = i_start
        while i < i_end:
            j = j_start
            while j < j_end:
                # does this cell already contain an observed sample?
                has_point = any(
                    i <= x < i + self.grid_size and
                    j <= y < j + self.grid_size
                    for (x, y, _, _) in self.actual_points
                )

                # empty cell: place a virtual point at its centre
                if not has_point:
                    virtual_x = i + self.grid_size / 2
                    virtual_y = j + self.grid_size / 2
                    virtual_points.append((virtual_x, virtual_y, 0.0, -1))  # -1: automatic virtual point

                j += self.grid_size
            i += self.grid_size

        # append the pre-set manual virtual points
        if self.manual_virtual_points:
            for point in self.manual_virtual_points:
                if len(point) == 2:
                    virtual_points.append((point[0], point[1], 0.0, -1))
                elif len(point) == 3:
                    virtual_points.append((point[0], point[1], point[2], -1))
                elif len(point) == 4:
                    virtual_points.append(point)
            print(f"Added {len(self.manual_virtual_points)} preset virtual points")

        # ====== interactive canvas ======
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        grid_count = (len(x_ticks) - 1) * (len(y_ticks) - 1)
        auto_virtual_count = len(virtual_points)
        print(f"Total cells: {grid_count}, automatic virtual points: {auto_virtual_count}")

        fig, ax = plt.subplots(figsize=(10, 8))

        # grid lines
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        # observed samples
        ax.scatter(x_coords, y_coords, c='red', s=20, label="Observed samples", zorder=3)
        ax.set_aspect('equal')
        # automatic virtual points
        if virtual_points:
            auto_vx = [p[0] for p in virtual_points]
            auto_vy = [p[1] for p in virtual_points]
            ax.scatter(auto_vx, auto_vy, c='blue', s=10, marker='x', label="Automatic virtual points", zorder=2)

        # title and instructions
        ax.set_title(
            "Grid layout\nLeft-click a vacant cell to add a zero-valued point\nClose the window to continue",
            color='blue',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')

        # === click handler ===
        def on_click(event):
            # accept left clicks inside the axes only
            if event.inaxes == ax and event.button == 1:
                # data coordinates of the click
                new_x, new_y = event.xdata, event.ydata

                # new zero-valued point (-2 marks an interactively added point)
                new_point = (new_x, new_y, 0.0, -2)

                # keep it on the instance
                self.manual_points.append(new_point)

                # draw it immediately
                ax.scatter(new_x, new_y, c='green', s=30, marker='+',
                           label="_nolegend_", zorder=4)
                ax.text(new_x, new_y, " 0", fontsize=6, color='green')

                # redraw the canvas
                fig.canvas.draw()
                print(f"Added zero-valued point: ({new_x:.2f}, {new_y:.2f})")

        # connect the handler
        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        # block until the user closes the window
        if self.interactive:
            plt.show()
        else:
            plt.close(fig)

        # merge automatic and interactive virtual points
        self.virtual_points = virtual_points + self.manual_points

        # return all virtual points together with the grid ticks
        print(
            f"Interaction finished. Automatic: {auto_virtual_count}, interactive: {len(self.manual_points)}, total virtual points: {len(self.virtual_points)}")

        return self.virtual_points, x_ticks, y_ticks

    def visualize_grid(self):
        """Draw the grid and report the number of cells."""
        if not self.actual_points:
            print("No observed samples; cannot draw the grid")
            return

        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # pad the bounding box
        i_start = min_x - self.grid_size
        i_end = max_x + self.grid_size
        j_start = min_y - self.grid_size
        j_end = max_y + self.grid_size

        # build the grid
        x_ticks = np.arange(i_start, i_end + self.grid_size, self.grid_size)
        y_ticks = np.arange(j_start, j_end + self.grid_size, self.grid_size)

        print(f"xgrid: {x_ticks}")
        print(f"ygrid: {y_ticks}")

        # count the cells
        grid_count = (len(x_ticks) - 1) * (len(y_ticks) - 1)
        print(f"Total cells: {grid_count}")

        # draw grid lines
        fig, ax = plt.subplots(figsize=(8, 6))
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in y_ticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        # observed samples
        ax.scatter(x_coords, y_coords, c='red', s=15, label="Actual Points")
        ax.set_title(f"Grid layout (total cells: {grid_count})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        if self.interactive:
            plt.show()

        return grid_count

    @staticmethod
    def calculate_distance(point1, point2):
        """Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def is_point_in_region(x, y, polygon):
        """Test whether a point lies inside a polygon."""
        point = Point(x, y)
        poly = Polygon(polygon)
        return poly.contains(point)

    def line_to_polygon(self, contour_set):
        """Convert a contour line into polygon regions."""
        face_polygons = []
        for path in contour_set.get_paths():
            paths = path.to_polygons()
            if paths:
                for path in paths:
                    face_polygons.append(path)
        return face_polygons

    def generate_contour_areas(self):
        """Build contour regions and their containment hierarchy."""
        # generate virtual points
        virtual_points, x_ticks, y_ticks = self.generate_virtual_points()

        all_points = self.actual_points + self.virtual_points

        # assemble the interpolation data
        x1 = np.array([p[0] for p in all_points])
        y1 = np.array([p[1] for p in all_points])
        z1 = np.array([p[2] for p in all_points])

        X, Y = np.meshgrid(x_ticks, y_ticks)

        # interpolate onto the regular grid
        zi = griddata((x1, y1), z1, (X, Y), method='linear')
        zi = np.nan_to_num(zi, nan=0)

        # extract the contours
        # larger figure and higher dpi for legible output
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')  # equal aspect ratio
        ax.tick_params(axis='both', which='major', labelsize=8)
        contour_set = ax.contour(X, Y, zi, levels=10, colors='black')

        # show the contours
        # ax.set_title('Contour Map with Actual and Virtual Points')

        # observed samples
        x_coords = [p[0] for p in self.actual_points]
        y_coords = [p[1] for p in self.actual_points]
        ax.scatter(x_coords, y_coords, c='red', s=15, zorder=4)

        # virtual points
        # if self.virtual_points:
        #     # distinguish automatic from interactive points
        #     auto_points = [p for p in self.virtual_points if p[3] == -1]
        #     manual_points = [p for p in self.virtual_points if p[3] == -2]
        #
        #     if auto_points:
        #         auto_vx = [p[0] for p in auto_points]
        #         auto_vy = [p[1] for p in auto_points]
        #         ax.scatter(auto_vx, auto_vy, c='blue', s=30, marker='x',
        #                    label="Auto Virtual Points", zorder=2)
        #
        #     if manual_points:
        #         manual_vx = [p[0] for p in manual_points]
        #         manual_vy = [p[1] for p in manual_points]
        #         ax.scatter(manual_vx, manual_vy, c='green', s=50, marker='+',
        #                    label="Manual Virtual Points", zorder=2)

        # ax.legend()
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.tight_layout()
        # save the figure
        plt.savefig('contour_areas.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

        if self.interactive:
            plt.show()
        else:
            plt.close(fig)

        # convert to polygons
        face_polygons = self.line_to_polygon(contour_set)

        # wrap into ContourArea objects
        contour_areas = []
        for polygon in face_polygons:
            points_in_poly = []
            for point in self.actual_points:
                if self.is_point_in_region(point[0], point[1], polygon):
                    points_in_poly.append(point)
            if points_in_poly:  # keep only regions that contain points
                contour_areas.append(self.ContourArea(polygon, points_in_poly))

        self.contour_areas = contour_areas

        # build the containment hierarchy
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
        """Return the density centres, i.e. the outermost contours."""
        # keep the outermost rings, which act as density centres
        outCircle = [area for area in self.contour_areas if area.parent is None]
        self.density_centers = outCircle
        return outCircle

    def assign_points_to_centers(self):
        """Assign points to density centres by expanding buffers."""
        if not self.density_centers:
            return {}

        point_assignments = {}
        assigned_points = set()

        # expand the buffer step by step
        for buffer_dist in range(0, self.max_buffer_distance + 1):
            current_assignments = {}

            # buffer for each density centre
            for center in self.density_centers:
                center_poly = Polygon(center.polygon)
                buffer_zone = center_poly.buffer(buffer_dist)

                # points inside the buffer
                for point in self.actual_points:
                    if point in assigned_points:
                        continue

                    pt = Point(point[0], point[1])
                    if buffer_zone.contains(pt):
                        if point not in current_assignments:
                            current_assignments[point] = []
                        current_assignments[point].append(center)

            # resolve overlaps when several buffers contain the same point
            for point, centers in current_assignments.items():
                if len(centers) > 1:
                    # keep the nearest centre
                    min_dist = float('inf')
                    closest_center = None
                    for center in centers:
                        dist = self.calculate_distance(point, center.center)
                        if dist < min_dist:
                            min_dist = dist
                            closest_center = center
                    current_assignments[point] = [closest_center]

                # record the final assignment
                self.point_to_density_center[point] = current_assignments[point]
                assigned_points.add(point)

        return self.point_to_density_center

    def assign_unassigned_points(self):
        """
        Assign unassigned points to their nearest cluster.
        For every point left unassigned, find the closest point in a cluster and adopt its label.

        Returns:
        assigned_count: number of points that were assigned
        """
        if not self.density_centers:
            print("Warning: no density centres; cannot assign unassigned points")
            return 0

        # points already assigned
        assigned_points = set(self.point_to_density_center.keys())

        # points not assigned to any centre
        unassigned_points = [point for point in self.actual_points
                             if point not in assigned_points]

        if not unassigned_points:
            print("All points are already assigned")
            return 0

        print(f"\nFound {len(unassigned_points)} unassigned points")

        # collect the clusters and their points
        cluster_points_map = {}
        for center in self.density_centers:
            cluster_points = [point for point, centers in self.point_to_density_center.items()
                              if center in centers]
            if cluster_points:  # ignore empty clusters
                cluster_points_map[center] = cluster_points

        if not cluster_points_map:
            print("Warning: no non-empty cluster available")
            return 0

        assigned_count = 0

        # nearest cluster for each unassigned point
        for unassigned_point in unassigned_points:
            min_dist = float('inf')
            nearest_center = None

            # scan all clusters for the closest member point
            for center, cluster_points in cluster_points_map.items():
                # distance to the closest member of the cluster
                for cluster_point in cluster_points:
                    dist = self.calculate_distance(unassigned_point, cluster_point)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_center = center

            if nearest_center is None:
                print(f"  Warning: point {unassigned_point[:2]} has no candidate cluster; left unassigned")
                continue

            # assign it to the nearest cluster
            self.point_to_density_center[unassigned_point] = [nearest_center]
            assigned_count += 1

        print(f"Assigned {assigned_count} unassigned points to their nearest cluster")
        return assigned_count

    def merge_small_clusters(self):
        """
        Merge small clusters into the nearest large one.
        Only clusters with at least min_cluster_size points are eligible targets.
        Each point of a small cluster moves to the large cluster holding its nearest neighbour.

        Returns:
        merged_count: number of merged small clusters
        """
        if not self.point_to_density_center or not self.density_centers:
            print("Warning: no clustering result; nothing to merge")
            return 0

        # initial point count per density centre
        center_point_counts = {}
        for center in self.density_centers:
            count = sum(1 for point, centers in self.point_to_density_center.items()
                        if center in centers)
            center_point_counts[center] = count

        # small clusters (fewer than min_cluster_size points)
        small_clusters = [center for center, count in center_point_counts.items()
                          if count < self.min_cluster_size and count > 0]

        if not small_clusters:
            print(f"No cluster has fewer than {self.min_cluster_size} points; nothing to merge")
            return 0

        print(f"\nFound {len(small_clusters)} small clusters (< {self.min_cluster_size} points)")

        # large clusters and their points
        large_clusters = {}
        for center, count in center_point_counts.items():
            if count >= self.min_cluster_size:
                # member points of this large cluster
                cluster_points = [point for point, centers in self.point_to_density_center.items()
                                  if center in centers]
                large_clusters[center] = cluster_points

        if not large_clusters:
            print(f"Warning: no cluster has at least {self.min_cluster_size} points; results unchanged")
            return 0

        print(f"Found {len(large_clusters)} large clusters (>= {self.min_cluster_size} points)")

        merged_count = 0
        total_reassigned_points = 0

        # move every point of every small cluster
        for small_center in small_clusters:
            # member points of the small cluster
            small_cluster_points = [point for point, centers in self.point_to_density_center.items()
                                    if small_center in centers]

            if not small_cluster_points:
                continue

            print(f"\n  Processing small cluster (centre: {small_center.center}, points: {len(small_cluster_points)})")

            # nearest large cluster for each point
            reassigned_points = 0
            for small_point in small_cluster_points:
                min_dist = float('inf')
                nearest_large_center = None

                # scan all large clusters for the closest member
                for large_center, large_points in large_clusters.items():
                    # distance to the closest member of the large cluster
                    for large_point in large_points:
                        dist = self.calculate_distance(small_point, large_point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_large_center = large_center

                if nearest_large_center is None:
                    print(f"    Warning: point {small_point[:2]} has no eligible large cluster; kept as is")
                    continue

                # reassign it to the nearest large cluster
                self.point_to_density_center[small_point] = [nearest_large_center]
                reassigned_points += 1

            if reassigned_points > 0:
                print(f"    Reassigned {reassigned_points} points to the nearest large cluster")
                merged_count += 1
                total_reassigned_points += reassigned_points

        print(f"\nMerged {merged_count} small clusters; {total_reassigned_points} points reassigned")
        return merged_count

    def mark_small_clusters_as_noise(self):
        """
        Mark small clusters as noise.
        Points of clusters with fewer than min_cluster_size members are dropped from the result.

        Returns:
        noise_count: number of points marked as noise
        """
        if not self.point_to_density_center or not self.density_centers:
            print("Warning: no clustering result; nothing to mark")
            return 0

        # point count per density centre
        center_point_counts = {}
        for center in self.density_centers:
            count = sum(1 for point, centers in self.point_to_density_center.items()
                        if center in centers)
            center_point_counts[center] = count

        # small clusters (fewer than min_cluster_size points)
        small_clusters = [center for center, count in center_point_counts.items()
                          if self.min_cluster_size > count > 0]

        if not small_clusters:
            print(f"No cluster has fewer than {self.min_cluster_size} points; no noise to mark")
            return 0

        print(f"\nFound {len(small_clusters)} small clusters (< {self.min_cluster_size} points); marking them as noise")

        noise_count = 0

        # drop the points of each small cluster
        for small_center in small_clusters:
            # member points of the small cluster
            small_cluster_points = [point for point, centers in self.point_to_density_center.items()
                                    if small_center in centers]

            if not small_cluster_points:
                continue

            print(f"  Small cluster (centre: {small_center.center}, points: {len(small_cluster_points)}) -> marked as noise")

            # removed here; get_predicted_labels() reports them as -1
            for point in small_cluster_points:
                if point in self.point_to_density_center:
                    del self.point_to_density_center[point]
                    noise_count += 1

        print(f"\nMarked {noise_count} points as noise")
        return noise_count

    def visualize_clusters(self):
        """Visualise the clustering result."""
        if not self.density_centers:
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')  # equal aspect ratio
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta',
                  'brown', 'pink', 'gray', 'olive', 'navy']

        # point count per centre
        center_point_counts = {}
        for center in self.density_centers:
            count = sum(1 for point, centers in self.point_to_density_center.items()
                        if center in centers)
            center_point_counts[center] = count

        # density centres and their buffers
        for i, center in enumerate(self.density_centers):
            point_count = center_point_counts.get(center, 0)
            if point_count == 0:
                continue  # skip empty clusters

            color = colors[i % len(colors)]

            # underlying contour region
            poly_patch = patches.Polygon(
                center.polygon,
                closed=True,
                fill=None,
                edgecolor=color,
                linestyle='dashed'
            )
            ax.add_patch(poly_patch)

            # observed samples of this centre
            for point, centers in self.point_to_density_center.items():
                if center in centers:
                    ax.plot(point[0], point[1], 'o', color=color, markersize=3)

            # cluster size
            ax.text(center.center[0], center.center[1], f'n={point_count}',
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # unassigned points
        all_assigned = {point for point in self.point_to_density_center}
        unassigned_count = 0
        for point in self.actual_points:
            if point not in all_assigned:
                ax.plot(point[0], point[1], 'ko', markersize=3)
                unassigned_count += 1

        # figure cosmetics
        title = 'Contour-Based Clustering Result'
        if self.min_cluster_size > 1:
            title += f' (min_cluster_size={self.min_cluster_size})'
        if unassigned_count > 0:
            title += f'\nUnassigned points: {unassigned_count}'

        # ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        # save the figure
        plt.savefig('cluster_result.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

        if self.interactive:
            plt.show()
        else:
            plt.close(fig)

    def calculate_inertia(self, X, labels):
        """Within-cluster sum of squares (inertia)."""
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
        """Return the per-point cluster labels."""
        final_clusters_hashable = {}
        for point, region_list in self.point_to_density_center.items():
            if all(isinstance(item, (int, float, str, tuple)) for item in region_list):
                final_clusters_hashable[point] = tuple(region_list)
            else:
                final_clusters_hashable[point] = tuple(id(item) for item in region_list)

        # distinct region labels
        regions = sorted(set(final_clusters_hashable.values()))
        region_to_label = {region: i for i, region in enumerate(regions)}

        # label every point; noise points get -1
        labels = []
        for point in self.actual_points:
            if tuple(point) in final_clusters_hashable:
                labels.append(region_to_label[final_clusters_hashable[tuple(point)]])
            else:
                labels.append(-1)

        return labels

    def silhouette_score(self):
        """Silhouette coefficient."""
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

        print("Contour clustering silhouette:", score)
        print("Contour clustering inertia:", inertia)

        return score, inertia

    def perform_clustering(self):
        """Run the full pipeline: clustering, small-cluster handling and assignment of unassigned points."""
        self.generate_contour_areas()
        self.get_density_centers()
        self.assign_points_to_centers()


        if self.merge_small_clusters_flag:  # True: merge into the nearest large cluster
            self.merge_small_clusters()
            self.assign_unassigned_points()
        else:  # False: mark the points of small clusters as noise
            self.mark_small_clusters_as_noise()

        # # small-cluster handling
        # if self.min_cluster_size > 1:
        #     if self.merge_small_clusters_flag:
        #     # True: merge into the nearest large cluster
        #         self.merge_small_clusters()
        #     # then assign the unassigned points
        #         self.assign_unassigned_points()
        #     else:
        #     # False: mark small clusters as noise
        #         self.mark_small_clusters_as_noise()
        #     # note: in noise mode unassigned points stay unassigned and become -1
        # else:
        #     # if min_cluster_size <= 1, only assign the unassigned points
        #     self.assign_unassigned_points()

        return self.point_to_density_center

    def get_cluster_report(self):
        """Return a summary report of the clustering result."""
        report = {}
        for center in self.density_centers:
            cluster_points = [
                point for point, centers in self.point_to_density_center.items()
                if center in centers
            ]
            if not cluster_points:  # skip empty clusters
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


# Example usage
if __name__ == "__main__":
    # observed samples (x, y, load, client id)
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
        # isolated points that form small clusters
        (15.0, 15.0, 1.0, 10),
        (15.5, 15.2, 1.2, 11),
        (16.0, 15.5, 0.8, 12),
    ]

    # manual virtual points (optional)
    manual_virtual_points = [
        (5.0, 5.0),
        (6.0, 6.0, 0.5),
        (7.0, 7.0, 0.3, -2),
    ]

    print("=" * 60)
    print("Example 1: merge_small_clusters=True (small clusters merged)")
    print("=" * 60)

    # instantiate with min_cluster_size=5 and merging enabled
    cluster1 = ContourBasedClusteringWithoutZ(
        actual_points,
        grid_size=1.0,
        max_buffer_distance=5,
        manual_virtual_points=manual_virtual_points,
        min_cluster_size=5,
        merge_small_clusters=True  # merge small clusters into the nearest large one
    )

    # run
    clusters1 = cluster1.perform_clustering()
    labels1 = cluster1.get_predicted_labels()
    print(f"Labels: {labels1}")
    print(f"Noise points: {labels1.count(-1)}")

    # visualise
    cluster1.visualize_clusters()

    print("\n" + "=" * 60)
    print("Example 2: merge_small_clusters=False (small clusters marked as noise)")
    print("=" * 60)

    # instantiate with min_cluster_size=5 and noise marking enabled
    cluster2 = ContourBasedClusteringWithoutZ(
        actual_points,
        grid_size=1.0,
        max_buffer_distance=5,
        manual_virtual_points=manual_virtual_points,
        min_cluster_size=5,
        merge_small_clusters=False  # mark small clusters as noise
    )

    # run
    clusters2 = cluster2.perform_clustering()
    labels2 = cluster2.get_predicted_labels()
    print(f"Labels: {labels2}")
    print(f"Noise points: {labels2.count(-1)}")

    # visualise
    cluster2.visualize_clusters()

    # summary report
    report = cluster2.get_cluster_report()
    print("\n" + "=" * 60)
    print("Clustering report (merge_small_clusters=False)")
    print("=" * 60)
    for center, data in report.items():
        print(f"\nCluster centre at {center}:")
        print(f"  Points: {data['point_count']}")
        print(f"  Total load: {data['total_load']:.2f}")
        print("  Members:")
        for point in data['points']:
            print(f"    Client {point[3]} at ({point[0]}, {point[1]}) load {point[2]}")
    print("=" * 60)
