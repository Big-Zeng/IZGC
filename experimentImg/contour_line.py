import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# --- 数据与设置 ---
grid_size = 4
x_coords = np.arange(grid_size + 1)
y_coords = np.arange(grid_size + 1)
Z = np.array([
    [1.0, 0.8, 0.5, 1.1, 1.2],
    [0.9, -0.5, -0.8, -0.3, 0.7],
    [0.6, -0.7, -2.0, -0.6, 0.4],
    [1.3, -0.4, -0.9, -0.2, 1.0],
    [1.1, 0.6, 0.7, 0.9, 1.3]
])
v_target = 0.0

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

def get_intersection(p1_coord, p2_coord, v1_val, v2_val, target_val):
    if (v1_val < target_val and v2_val < target_val) or \
       (v1_val >= target_val and v2_val >= target_val):
        return None
    ratio = (target_val - v1_val) / (v2_val - v1_val)
    return p1_coord + ratio * (p2_coord - p1_coord)

# --- 图 (a): 初始网格点值 ---
def save_plot_a():
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            cell = Polygon([[j, i], [j+1, i], [j+1, i+1], [j, i+1]],
                           closed=True, fill=False, edgecolor='lightgray', lw=0.5)
            ax.add_patch(cell)

    for r_idx in range(grid_size + 1):
        for c_idx in range(grid_size + 1):
            val = Z[r_idx, c_idx]
            color = 'red' if val < v_target else 'blue'
            ax.scatter(c_idx, r_idx, c=color, s=100, edgecolors='black', zorder=5)
            ax.text(c_idx, r_idx + 0.15, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=color)

    ax.set_title("(a) Initial Grid Field Values", fontsize=14)
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('plot_a_grid_values.png', dpi=300)
    plt.close()

# --- 图 (b): 局部边插值与线段生成 ---
def save_plot_b():
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(grid_size):
        for j in range(grid_size):
            # 1. 绘制网格
            cell = Polygon([[j, i], [j+1, i], [j+1, i+1], [j, i+1]],
                           closed=True, fill=False, edgecolor='lightgray', lw=0.5)
            ax.add_patch(cell)

            # 2. 获取顶点数据
            coords = [np.array([j, i]), np.array([j+1, i]), np.array([j+1, i+1]), np.array([j, i+1])]
            vals = [Z[i, j], Z[i, j+1], Z[i+1, j+1], Z[i+1, j]]

            # 3. 绘制顶点
            for k in range(4):
                color = 'red' if vals[k] < v_target else 'blue'
                ax.scatter(coords[k][0], coords[k][1], c=color, s=100, edgecolors='black', zorder=5)

            # 4. 计算并绘制插值交点与线段
            intersections = []
            # 依次检查四条边: 底, 右, 顶, 左
            edges = [(0,1), (1,2), (2,3), (3,0)]
            for e1, e2 in edges:
                p = get_intersection(coords[e1], coords[e2], vals[e1], vals[e2], v_target)
                if p is not None: intersections.append(p)

            if len(intersections) == 2:
                ax.scatter([p[0] for p in intersections], [p[1] for p in intersections],
                           c='green', marker='x', s=120, lw=2.5, zorder=6)
                ax.plot([intersections[0][0], intersections[1][0]], [intersections[0][1], intersections[1][1]],
                        color='green', linewidth=3, zorder=4)
            elif len(intersections) == 4: # 鞍点
                ax.scatter([p[0] for p in intersections], [p[1] for p in intersections],
                           c='green', marker='x', s=120, lw=2.5, zorder=6)
                ax.plot([intersections[0][0], intersections[2][0]], [intersections[0][1], intersections[2][1]],
                        color='green', linewidth=3, zorder=4)
                ax.plot([intersections[3][0], intersections[1][0]], [intersections[3][1], intersections[1][1]],
                        color='green', linewidth=3, zorder=4)

    ax.set_title("(b) Local Interpolation & Segment Generation", fontsize=14)
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('plot_b_interpolation.png', dpi=300)
    plt.close()

# 执行生成
save_plot_a()
save_plot_b()
print("已成功分别生成 plot_a_grid_values.png 和 plot_b_interpolation.png")