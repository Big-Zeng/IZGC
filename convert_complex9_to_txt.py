"""
将data/complex9.csv转换为txt格式（空格分隔），并将target值+1
"""

import pandas as pd
import numpy as np

# 读取CSV文件

df = pd.read_csv('data/2d_3c.csv')

print(f"✓ 读取成功，共 {len(df)} 个点")

# 提取数据
x = df['x'].values
y = df['y'].values
target = df['target'].values

print(f"  原始target范围: [{target.min()}, {target.max()}]")
print(f"  原始簇数: {len(np.unique(target))}")

# 将target值+1
target_new = target + 1

print(f"  新target范围: [{target_new.min()}, {target_new.max()}]")
print(f"  新簇数: {len(np.unique(target_new))}")

# 保存为txt文件（空格分隔）
output_file = 'data/2d_3c.txt'
with open(output_file, 'w') as f:
    for i in range(len(x)):
        f.write(f"{x[i]} {y[i]} {target_new[i]}\n")

print(f"\n✓ 已保存为 {output_file} (空格分隔)")

# 显示前几行
print("\n前5行数据:")
for i in range(min(5, len(x))):
    print(f"{x[i]} {y[i]} {target_new[i]}")

print("\n数据统计:")
print(f"  X范围: [{x.min():.2f}, {x.max():.2f}]")
print(f"  Y范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"  Target范围: [{target_new.min()}, {target_new.max()}]")
print(f"  总点数: {len(x)}")
print(f"  簇数: {len(np.unique(target_new))}")
