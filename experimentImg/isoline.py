import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. Basic parameter settings
np.random.seed(42)
grid_size = 10  # Delta_g
R = 15          # Density search radius

# 2. Generate the raw points, grouped into two clusters
cluster1 = np.random.randn(15, 2) * 8 + [30, 30]
cluster2 = np.random.randn(15, 2) * 8 + [70, 70]
points = np.vstack([cluster1, cluster2])

# 3. Compute the point density Z from the search radius R
def calculate_density(p, all_pts, radius):
    dists = np.linalg.norm(all_pts - p, axis=1)
    return np.sum(dists <= radius)

z_points = np.array([calculate_density(p, points, R) for p in points])

# 4. Dynamic zero-point injection, extending two grid steps only
# Compute the bounding box of the points
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

# Define the extended range
margin = 2 * grid_size
x_start, x_end = x_min - margin, x_max + margin
y_start, y_end = y_min - margin, y_max + margin

# Generate the cell centres covering this range
x_range = np.arange(x_start, x_end + grid_size, grid_size)
y_range = np.arange(y_start, y_end + grid_size, grid_size)
gx, gy = np.meshgrid(x_range, y_range)
grid_centers = np.vstack([gx.ravel(), gy.ravel()]).T

zero_points = []
for cp in grid_centers:
    # Rule: inject a zero point into every cell that holds no data point
    in_cell = (points[:, 0] >= cp[0] - grid_size/2) & (points[:, 0] < cp[0] + grid_size/2) & \
              (points[:, 1] >= cp[1] - grid_size/2) & (points[:, 1] < cp[1] + grid_size/2)
    if not np.any(in_cell):
        zero_points.append(cp)

zero_points = np.array(zero_points)
z_zeros = np.zeros(len(zero_points))

# Merge the augmented point set
all_coords = np.vstack([points, zero_points])
all_z = np.concatenate([z_points, z_zeros])

# 5. Publication-style figure settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- Fig. 1: distribution of data points and zero points ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.scatter(zero_points[:, 0], zero_points[:, 1], c='gray', marker='x', s=25, alpha=0.5)
ax1.scatter(points[:, 0], points[:, 1], c='#1f77b4', s=50, edgecolors='k', zorder=5)

ax1.set_xlim(x_start - 5, x_end + 5)
ax1.set_ylim(y_start - 5, y_end + 5)
ax1.axis('on')
plt.savefig('izgc_distribution_tight.png', dpi=300, bbox_inches='tight')

# --- Fig. 2: extracted contours (cluster boundaries) ---
fig2, ax2 = plt.subplots(figsize=(7, 7))
xi = np.linspace(x_start - 5, x_end + 5, 200)
yi = np.linspace(y_start - 5, y_end + 5, 200)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata(all_coords, all_z, (Xi, Yi), method='linear')
Zi = np.nan_to_num(Zi, nan=0.0)

# levels=3 draws three contours; set [0.5] to keep the boundary only
ax2.contour(Xi, Yi, Zi, levels=3, colors='green', linewidths=1.2)
ax2.scatter(points[:, 0], points[:, 1], c='#1f77b4', s=50, edgecolors='k', zorder=3)

ax2.set_xlim(x_start - 5, x_end + 5)
ax2.set_ylim(y_start - 5, y_end + 5)
ax2.axis('on')
plt.savefig('izgc_isolines_tight.png', dpi=300, bbox_inches='tight')

print("动态延伸 2 个 grid_size 的图表已生成。")