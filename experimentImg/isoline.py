import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. 基础参数设置
np.random.seed(42)
grid_size = 10  # Delta_g
R = 15          # 密度搜索半径

# 2. 生成原始离散点 (聚集成两个簇)
cluster1 = np.random.randn(15, 2) * 8 + [30, 30]
cluster2 = np.random.randn(15, 2) * 8 + [70, 70]
points = np.vstack([cluster1, cluster2])

# 3. 计算离散点密度 Z (基于搜索半径 R)
def calculate_density(p, all_pts, radius):
    dists = np.linalg.norm(all_pts - p, axis=1)
    return np.sum(dists <= radius)

z_points = np.array([calculate_density(p, points, R) for p in points])

# 4. 动态零点注入逻辑 (仅延伸 2 个 grid_size)
# 计算离散点的包围盒
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

# 定义延伸后的范围
margin = 2 * grid_size
x_start, x_end = x_min - margin, x_max + margin
y_start, y_end = y_min - margin, y_max + margin

# 生成覆盖该范围的网格中心
x_range = np.arange(x_start, x_end + grid_size, grid_size)
y_range = np.arange(y_start, y_end + grid_size, grid_size)
gx, gy = np.meshgrid(x_range, y_range)
grid_centers = np.vstack([gx.ravel(), gy.ravel()]).T

zero_points = []
for cp in grid_centers:
    # 判定规则：网格单元内无离散点则注入零点
    in_cell = (points[:, 0] >= cp[0] - grid_size/2) & (points[:, 0] < cp[0] + grid_size/2) & \
              (points[:, 1] >= cp[1] - grid_size/2) & (points[:, 1] < cp[1] + grid_size/2)
    if not np.any(in_cell):
        zero_points.append(cp)

zero_points = np.array(zero_points)
z_zeros = np.zeros(len(zero_points))

# 合并增强点集
all_coords = np.vstack([points, zero_points])
all_z = np.concatenate([z_points, z_zeros])

# 5. 学术绘图设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- 图 1: 离散点与零值点分布 ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.scatter(zero_points[:, 0], zero_points[:, 1], c='gray', marker='x', s=25, alpha=0.5)
ax1.scatter(points[:, 0], points[:, 1], c='#1f77b4', s=50, edgecolors='k', zorder=5)

ax1.set_xlim(x_start - 5, x_end + 5)
ax1.set_ylim(y_start - 5, y_end + 5)
ax1.axis('on')
plt.savefig('izgc_distribution_tight.png', dpi=300, bbox_inches='tight')

# --- 图 2: 生成的等值线 (聚类边界) ---
fig2, ax2 = plt.subplots(figsize=(7, 7))
xi = np.linspace(x_start - 5, x_end + 5, 200)
yi = np.linspace(y_start - 5, y_end + 5, 200)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata(all_coords, all_z, (Xi, Yi), method='linear')
Zi = np.nan_to_num(Zi, nan=0.0)

# levels=3 会绘制三条等高线，如需只看边界可设为 [0.5]
ax2.contour(Xi, Yi, Zi, levels=3, colors='green', linewidths=1.2)
ax2.scatter(points[:, 0], points[:, 1], c='#1f77b4', s=50, edgecolors='k', zorder=3)

ax2.set_xlim(x_start - 5, x_end + 5)
ax2.set_ylim(y_start - 5, y_end + 5)
ax2.axis('on')
plt.savefig('izgc_isolines_tight.png', dpi=300, bbox_inches='tight')

print("动态延伸 2 个 grid_size 的图表已生成。")