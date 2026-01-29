import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.patches import Polygon

# 1. 数据准备
np.random.seed(42)
# 生成原始样本点 (TIN的顶点)
points = np.random.rand(30, 2) * 100
tri = Delaunay(points)

# 生成规则网格 (Grid Points)
x_grid = np.linspace(0, 100, 40)
y_grid = np.linspace(0, 100, 40)
GX, GY = np.meshgrid(x_grid, y_grid)
grid_points = np.vstack([GX.ravel(), GY.ravel()]).T

# 2. 筛选演示对象
center = np.array([50, 50])
simplex_id = tri.find_simplex(center)
vert_indices = tri.simplices[simplex_id]
A, B, C = points[vert_indices[0]], points[vert_indices[1]], points[vert_indices[2]]


# 判断点是否在三角形内
def is_inside(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


target_P = None
triangle_centroid = (A + B + C) / 3
min_dist = 1e9
for p in grid_points:
    if is_inside(p, A, B, C):
        dist = np.linalg.norm(p - triangle_centroid)
        if dist < min_dist:
            min_dist = dist
            target_P = p

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(8, 8))

# --- 层级 1: 绘制全场网格背景 (Grid Background) ---
# 颜色改为更深一点的蓝色，如 'steelblue' 或 '#1f77b4' (Matplotlib 默认蓝)
# alpha 不变或稍微调高一点点，让它明显一些但还是背景
ax.scatter(GX, GY, s=15, c='#1f77b4', alpha=0.6, marker='.')

# --- 层级 2: 绘制 TIN 骨架 ---
ax.triplot(points[:, 0], points[:, 1], tri.simplices, c='gray', lw=0.5, alpha=0.5)

# --- 层级 3: 聚焦特定三角形 ---
tri_poly = Polygon([A, B, C], closed=True, fill=False, edgecolor='black', lw=2, zorder=5)
ax.add_patch(tri_poly)
ax.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], c='black', s=80, zorder=6)

# --- 层级 4: 绘制选中的网格点 P 及其插值区域 ---
if target_P is not None:
    # 绘制点 P
    ax.scatter(target_P[0], target_P[1], c='red', s=100, marker='x', lw=2, zorder=10, label='Selected Grid Point $P$')

    # 区域
    poly_A = Polygon([target_P, B, C], closed=True, color='#ffcccb', alpha=0.6)
    poly_B = Polygon([target_P, A, C], closed=True, color='#90ee90', alpha=0.6)
    poly_C = Polygon([target_P, A, B], closed=True, color='#add8e6', alpha=0.6)

    ax.add_patch(poly_A)
    ax.add_patch(poly_B)
    ax.add_patch(poly_C)

    # 虚线连接：加粗一点以便看清
    ax.plot([target_P[0], A[0]], [target_P[1], A[1]], 'k--', lw=1.2, alpha=0.8)
    ax.plot([target_P[0], B[0]], [target_P[1], B[1]], 'k--', lw=1.2, alpha=0.8)
    ax.plot([target_P[0], C[0]], [target_P[1], C[1]], 'k--', lw=1.2, alpha=0.8)

    # --- 层级 5: 标注 ---
    ax.text((target_P[0] + B[0] + C[0]) / 3, (target_P[1] + B[1] + C[1]) / 3, '$S_A$', fontsize=12, fontweight='bold',
            color='darkred', ha='center')
    ax.text((target_P[0] + A[0] + C[0]) / 3, (target_P[1] + A[1] + C[1]) / 3, '$S_B$', fontsize=12, fontweight='bold',
            color='darkgreen', ha='center')
    ax.text((target_P[0] + A[0] + B[0]) / 3, (target_P[1] + A[1] + B[1]) / 3, '$S_C$', fontsize=12, fontweight='bold',
            color='darkblue', ha='center')

    ax.text(A[0], A[1] + 1.5, '$A$', fontsize=12, fontweight='bold', ha='center')
    ax.text(B[0], B[1] - 2.5, '$B$', fontsize=12, fontweight='bold', ha='center')
    ax.text(C[0], C[1] - 2.5, '$C$', fontsize=12, fontweight='bold', ha='center')
    ax.text(target_P[0], target_P[1] + 1.5, '$P(x,y)$', fontsize=12, fontweight='bold', color='red', ha='center')


# 设置视图范围
margin = 15
ax.set_xlim(min(A[0], B[0], C[0]) - margin, max(A[0], B[0], C[0]) + margin)
ax.set_ylim(min(A[1], B[1], C[1]) - margin, max(A[1], B[1], C[1]) + margin)

# 装饰：去掉了 ax.grid()
ax.set_title("Grid Interpolation Mechanism: From TIN to Regular Grid", fontsize=14)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc='upper right')
# ax.grid(True) # 这一行被去掉了，也就是"一般的背景虚线给去掉"

plt.savefig('grid_interpolation_mechanism_300dpi_v2.png', dpi=300, bbox_inches='tight')
plt.show()