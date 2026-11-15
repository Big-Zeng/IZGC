"""
生成双环形聚类数据集
外圆环：400个点，标签为1
内圆环：200个点，标签为2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fix the random seed for reproducibility
np.random.seed(42)

# Parameter settings
n_outer = 400  # Number of points in the outer ring
n_inner = 200  # Number of points in the inner ring
center_x, center_y = 50, 50  # Centre coordinates

# Outer-ring parameters
outer_radius_mean = 30  # Mean radius of the outer ring
outer_radius_std = 2    # Radius standard deviation of the outer ring (controls density)

# Inner-ring parameters
inner_radius_mean = 15  # Mean radius of the inner ring
inner_radius_std = 1.5  # Radius standard deviation of the inner ring (denser)

# Generate the outer-ring data
outer_angles = np.random.uniform(0, 2 * np.pi, n_outer)
outer_radii = np.random.normal(outer_radius_mean, outer_radius_std, n_outer)
outer_x = center_x + outer_radii * np.cos(outer_angles)
outer_y = center_y + outer_radii * np.sin(outer_angles)
outer_labels = np.ones(n_outer, dtype=int)

# Generate the inner-ring data
inner_angles = np.random.uniform(0, 2 * np.pi, n_inner)
inner_radii = np.random.normal(inner_radius_mean, inner_radius_std, n_inner)
inner_x = center_x + inner_radii * np.cos(inner_angles)
inner_y = center_y + inner_radii * np.sin(inner_angles)
inner_labels = np.full(n_inner, 2, dtype=int)

# Merge the data
x = np.concatenate([outer_x, inner_x])
y = np.concatenate([outer_y, inner_y])
labels = np.concatenate([outer_labels, inner_labels])

# Shuffle the data order
indices = np.random.permutation(len(x))
x = x[indices]
y = y[indices]
labels = labels[indices]

# Save as CSV
df = pd.DataFrame({
    'x': x,
    'y': y,
    'target': labels
})
df.to_csv('data/ring.csv', index=False)
print(f"✓ 已生成 data/ring.csv ({len(df)} 个点)")

# Save as tab-separated TXT including labels
with open('data/ring.txt', 'w') as f:
    for i in range(len(x)):
        f.write(f"{x[i]:.12f}\t{y[i]:.12f}\t{labels[i]}\n")
print(f"✓ 已生成 data/ring.txt ({len(x)} 个点，包含标签)")

# Visualise the data
plt.figure(figsize=(10, 10))
plt.scatter(outer_x, outer_y, c='red', s=20, alpha=0.6, label=f'外圆环 (n={n_outer})')
plt.scatter(inner_x, inner_y, c='blue', s=20, alpha=0.6, label=f'内圆环 (n={n_inner})')
plt.scatter(center_x, center_y, c='green', s=100, marker='*', label='圆心')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'双环形聚类数据集 (总计 {n_outer + n_inner} 个点)')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.savefig('data/ring_visualization.png', dpi=150, bbox_inches='tight')
print("✓ 已生成可视化图片 data/ring_visualization.png")
plt.show()

# Print the statistics
print("\n" + "="*60)
print("数据集统计信息")
print("="*60)
print(f"总点数: {len(x)}")
print(f"外圆环点数: {n_outer} (标签=1)")
print(f"内圆环点数: {n_inner} (标签=2)")
print(f"\n外圆环:")
print(f"  平均半径: {outer_radius_mean}")
print(f"  半径标准差: {outer_radius_std}")
print(f"  X范围: [{outer_x.min():.2f}, {outer_x.max():.2f}]")
print(f"  Y范围: [{outer_y.min():.2f}, {outer_y.max():.2f}]")
print(f"\n内圆环:")
print(f"  平均半径: {inner_radius_mean}")
print(f"  半径标准差: {inner_radius_std}")
print(f"  X范围: [{inner_x.min():.2f}, {inner_x.max():.2f}]")
print(f"  Y范围: [{inner_y.min():.2f}, {inner_y.max():.2f}]")
print("="*60)
