import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 设置随机数种子以保证结果可复现
np.random.seed(10)

# 1. 生成随机点数据
# 点的数量，设置为30以保证不太密集
num_points = 15
# 生成 0-100 范围内的二维随机点
points = np.random.rand(num_points, 2) * 100

# 2. 生成 Delaunay 三角网 (TIN)
tri = Delaunay(points)

# 3. 创建绘图画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
# 设置全局字体（推荐使用字体，如Arial或Times New Roman）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- 图 1：随机分布点集 ---
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(points[:, 0], points[:, 1], color='#1f77b4', s=60, edgecolors='k', linewidth=0.5, zorder=3)
ax1.set_title("Discrete Sampling Points", fontsize=16)
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.set_xlabel("X ", fontsize=12)
ax1.set_ylabel("Y ", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.5)
# 保存为 300 DPI 的高清图
plt.savefig('scatter_points_300dpi.png', dpi=300, bbox_inches='tight')

# --- 图 2：生成的 TIN ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
# 绘制三角网边
ax2.triplot(points[:, 0], points[:, 1], tri.simplices, color='#7f7f7f', linewidth=1.2, zorder=1)
# 绘制顶点
ax2.scatter(points[:, 0], points[:, 1], color='#1f77b4', s=60, edgecolors='k', linewidth=0.5, zorder=3)
ax2.set_title("Generated TIN (Delaunay)", fontsize=16)
ax2.set_xlim(-5, 105)
ax2.set_ylim(-5, 105)
ax2.set_xlabel("X ", fontsize=12)
ax2.set_ylabel("Y ", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.5)
# 保存为 300 DPI 的高清图
plt.savefig('tin_network_300dpi.png', dpi=300, bbox_inches='tight')


# 显示图像
plt.show()