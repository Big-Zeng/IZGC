"""
生成 LineBlobs 聚类数据集
形状：一条弯曲的线形簇 + 两个密集的圆形簇
- 线形簇：标签=1，密度稀疏，呈S形或弧形
- 圆形簇1：标签=2，密度密集
- 圆形簇2：标签=3，密度密集
总共266个点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fix the random seed for reproducibility
np.random.seed(42)

# Parameter settings
n_line = 150       # Points in the linear cluster (numerous but sparse)
n_blob1 = 58       # Points in circular cluster 1
n_blob2 = 58       # Points in circular cluster 2
total_points = n_line + n_blob1 + n_blob2

print(f"目标总点数: 266")
print(f"实际总点数: {total_points}")

# ==================== Generate the linear cluster (label = 1) ====================
# Create an upward-opening half arc (U shape) enclosing the rectangles at a distance
arc_angles = np.linspace(np.pi * 0.15, np.pi * 0.85, n_line)  # From 27 to 153 degrees, a wider range
arc_radius = 50  # A larger arc radius, placing the arc further from the rectangles
arc_center_x, arc_center_y = 50, 10  # Lower the arc centre so that it opens upwards

line_x = arc_center_x + arc_radius * np.cos(arc_angles)
line_y = arc_center_y + arc_radius * np.sin(arc_angles)

# Add noise to make it sparser
noise_x = np.random.normal(0, 2.5, n_line)
noise_y = np.random.normal(0, 2.5, n_line)
line_x = line_x + noise_x
line_y = line_y + noise_y
line_labels = np.ones(n_line, dtype=int)

# ==================== Generate rectangle 1 (label = 2) ====================
# Placement: left rectangle, inside the half arc but well away from it
rect1_x_min, rect1_x_max = 28, 38
rect1_y_min, rect1_y_max = 35, 50

# Generate the points of rectangle 1 (dense)
rect1_x = np.random.uniform(rect1_x_min, rect1_x_max, n_blob1)
rect1_y = np.random.uniform(rect1_y_min, rect1_y_max, n_blob1)
rect1_labels = np.full(n_blob1, 2, dtype=int)

# ==================== Generate rectangle 2 (label = 3) ====================
# Placement: right rectangle, inside the half arc but well away from it
rect2_x_min, rect2_x_max = 62, 72
rect2_y_min, rect2_y_max = 35, 50

# Generate the points of rectangle 2 (dense)
rect2_x = np.random.uniform(rect2_x_min, rect2_x_max, n_blob2)
rect2_y = np.random.uniform(rect2_y_min, rect2_y_max, n_blob2)
rect2_labels = np.full(n_blob2, 3, dtype=int)

# ==================== Merge all data ====================
x = np.concatenate([line_x, rect1_x, rect2_x])
y = np.concatenate([line_y, rect1_y, rect2_y])
labels = np.concatenate([line_labels, rect1_labels, rect2_labels])

# Shuffle the data order
indices = np.random.permutation(len(x))
x = x[indices]
y = y[indices]
labels = labels[indices]

# ==================== Save the data ====================
# Save as CSV
df = pd.DataFrame({
    'x': x,
    'y': y,
    'target': labels
})
df.to_csv('data/LineBlobs.csv', index=False)
print(f"✓ 已生成 data/LineBlobs.csv ({len(df)} 个点)")

# Save as tab-separated TXT including labels
with open('data/LineBlobs.txt', 'w') as f:
    for i in range(len(x)):
        f.write(f"{x[i]:.12f}\t{y[i]:.12f}\t{labels[i]}\n")
print(f"✓ 已生成 data/LineBlobs.txt ({len(x)} 个点，包含标签)")

# ==================== Visualise the data ====================
plt.figure(figsize=(10, 10))

# Draw the half arc
plt.scatter(line_x, line_y, c='green', s=30, alpha=0.6, label=f'半弧形 (n={n_line}, 标签=1, 稀疏)')

# Draw rectangle 1
plt.scatter(rect1_x, rect1_y, c='red', s=30, alpha=0.6, label=f'矩形1 (n={n_blob1}, 标签=2, 密集)')

# Draw rectangle 2
plt.scatter(rect2_x, rect2_y, c='blue', s=30, alpha=0.6, label=f'矩形2 (n={n_blob2}, 标签=3, 密集)')

# Draw the rectangle outlines as guides
rect1_corners_x = [rect1_x_min, rect1_x_max, rect1_x_max, rect1_x_min, rect1_x_min]
rect1_corners_y = [rect1_y_min, rect1_y_min, rect1_y_max, rect1_y_max, rect1_y_min]
plt.plot(rect1_corners_x, rect1_corners_y, 'r--', alpha=0.3, linewidth=1)

rect2_corners_x = [rect2_x_min, rect2_x_max, rect2_x_max, rect2_x_min, rect2_x_min]
rect2_corners_y = [rect2_y_min, rect2_y_min, rect2_y_max, rect2_y_max, rect2_y_min]
plt.plot(rect2_corners_x, rect2_corners_y, 'b--', alpha=0.3, linewidth=1)

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title(f'LineBlobs Dataset (Total: {total_points} points)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/LineBlobs_visualization.png', dpi=150, bbox_inches='tight')
print("✓ 已生成可视化图片 data/LineBlobs_visualization.png")
plt.show()

# ==================== Print the statistics ====================
print("\n" + "="*60)
print("LineBlobs 数据集统计信息")
print("="*60)
print(f"总点数: {len(x)}")

print(f"\n半弧形 (标签=1):")
print(f"  点数: {n_line}")
print(f"  形状: 半弧形（包裹矩形）")
print(f"  中心: ({arc_center_x}, {arc_center_y})")
print(f"  半径: {arc_radius}")
print(f"  X范围: [{line_x.min():.2f}, {line_x.max():.2f}]")
print(f"  Y范围: [{line_y.min():.2f}, {line_y.max():.2f}]")
print(f"  密度: 稀疏（沿弧线分布）")

print(f"\n矩形1 (标签=2):")
print(f"  点数: {n_blob1}")
print(f"  X范围: [{rect1_x_min}, {rect1_x_max}]")
print(f"  Y范围: [{rect1_y_min}, {rect1_y_max}]")
print(f"  面积: {(rect1_x_max - rect1_x_min) * (rect1_y_max - rect1_y_min)}")
print(f"  密度: 密集 ({n_blob1 / ((rect1_x_max - rect1_x_min) * (rect1_y_max - rect1_y_min)):.2f} 点/单位面积)")

print(f"\n矩形2 (标签=3):")
print(f"  点数: {n_blob2}")
print(f"  X范围: [{rect2_x_min}, {rect2_x_max}]")
print(f"  Y范围: [{rect2_y_min}, {rect2_y_max}]")
print(f"  面积: {(rect2_x_max - rect2_x_min) * (rect2_y_max - rect2_y_min)}")
print(f"  密度: 密集 ({n_blob2 / ((rect2_x_max - rect2_x_min) * (rect2_y_max - rect2_y_min)):.2f} 点/单位面积)")

print("\n数据分布:")
print(f"  整体X范围: [{x.min():.2f}, {x.max():.2f}]")
print(f"  整体Y范围: [{y.min():.2f}, {y.max():.2f}]")
print("="*60)
