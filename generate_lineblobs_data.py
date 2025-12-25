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

# 设置随机种子以保证可重复性
np.random.seed(42)

# 参数设置
n_line = 150       # 线形簇点数（较多，但密度稀疏）
n_blob1 = 58       # 圆形簇1点数
n_blob2 = 58       # 圆形簇2点数
total_points = n_line + n_blob1 + n_blob2

print(f"目标总点数: 266")
print(f"实际总点数: {total_points}")

# ==================== 生成线形簇（标签=1）====================
# 创建一条向上开口的半弧形（U形），包裹矩形但距离较远
arc_angles = np.linspace(np.pi * 0.15, np.pi * 0.85, n_line)  # 从27度到153度，更宽
arc_radius = 50  # 更大的弧形半径，与矩形距离更远
arc_center_x, arc_center_y = 50, 10  # 弧形中心位置更低，使其向上开口

line_x = arc_center_x + arc_radius * np.cos(arc_angles)
line_y = arc_center_y + arc_radius * np.sin(arc_angles)

# 添加噪声使其更稀疏
noise_x = np.random.normal(0, 2.5, n_line)
noise_y = np.random.normal(0, 2.5, n_line)
line_x = line_x + noise_x
line_y = line_y + noise_y
line_labels = np.ones(n_line, dtype=int)

# ==================== 生成矩形1（标签=2）====================
# 位置：左侧矩形（在半弧内部，距离弧线较远）
rect1_x_min, rect1_x_max = 28, 38
rect1_y_min, rect1_y_max = 35, 50

# 生成矩形1点（密集分布）
rect1_x = np.random.uniform(rect1_x_min, rect1_x_max, n_blob1)
rect1_y = np.random.uniform(rect1_y_min, rect1_y_max, n_blob1)
rect1_labels = np.full(n_blob1, 2, dtype=int)

# ==================== 生成矩形2（标签=3）====================
# 位置：右侧矩形（在半弧内部，距离弧线较远）
rect2_x_min, rect2_x_max = 62, 72
rect2_y_min, rect2_y_max = 35, 50

# 生成矩形2点（密集分布）
rect2_x = np.random.uniform(rect2_x_min, rect2_x_max, n_blob2)
rect2_y = np.random.uniform(rect2_y_min, rect2_y_max, n_blob2)
rect2_labels = np.full(n_blob2, 3, dtype=int)

# ==================== 合并所有数据 ====================
x = np.concatenate([line_x, rect1_x, rect2_x])
y = np.concatenate([line_y, rect1_y, rect2_y])
labels = np.concatenate([line_labels, rect1_labels, rect2_labels])

# 打乱数据顺序
indices = np.random.permutation(len(x))
x = x[indices]
y = y[indices]
labels = labels[indices]

# ==================== 保存数据 ====================
# 保存为CSV格式
df = pd.DataFrame({
    'x': x,
    'y': y,
    'target': labels
})
df.to_csv('data/LineBlobs.csv', index=False)
print(f"✓ 已生成 data/LineBlobs.csv ({len(df)} 个点)")

# 保存为TXT格式（制表符分隔，包含标签）
with open('data/LineBlobs.txt', 'w') as f:
    for i in range(len(x)):
        f.write(f"{x[i]:.12f}\t{y[i]:.12f}\t{labels[i]}\n")
print(f"✓ 已生成 data/LineBlobs.txt ({len(x)} 个点，包含标签)")

# ==================== 可视化数据 ====================
plt.figure(figsize=(10, 10))

# 绘制半弧形
plt.scatter(line_x, line_y, c='green', s=30, alpha=0.6, label=f'半弧形 (n={n_line}, 标签=1, 稀疏)')

# 绘制矩形1
plt.scatter(rect1_x, rect1_y, c='red', s=30, alpha=0.6, label=f'矩形1 (n={n_blob1}, 标签=2, 密集)')

# 绘制矩形2
plt.scatter(rect2_x, rect2_y, c='blue', s=30, alpha=0.6, label=f'矩形2 (n={n_blob2}, 标签=3, 密集)')

# 绘制矩形边界（辅助线）
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

# ==================== 打印统计信息 ====================
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
