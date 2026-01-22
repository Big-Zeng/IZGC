"""
绘制Jain数据集的稳定性分析图
- 无标题、无图例
- ARI、AMI、FMI合并一张图
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'results_stability')

# 读取稀释后的数据
df = pd.read_csv(os.path.join(output_dir, 'stability_results_Jain_sparse.csv'))

# 只保留有效数据
df_valid = df[df['valid'] == True].copy()
print(f'有效数据行数: {len(df_valid)}')

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# ========== 图1: 簇数量变化（无红线）==========
fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
ax1.plot(df_valid['grid_size'], df_valid['n_clusters'], 'b-', linewidth=1.5, marker='o', markersize=3)
ax1.set_xlabel('grid_size')
ax1.set_ylabel('Number of Clusters')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([df_valid['grid_size'].min(), df_valid['grid_size'].max()])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stability_clusters_Jain.png'), dpi=300, bbox_inches='tight')
plt.close()
print('已保存: stability_clusters_Jain.png')

# ========== 图2: ARI、AMI、FMI 合并 ==========
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(df_valid['grid_size'], df_valid['ARI'], 'b-', linewidth=1.5, marker='o', markersize=2, label='ARI')
ax.plot(df_valid['grid_size'], df_valid['AMI'], 'g-', linewidth=1.5, marker='s', markersize=2, label='AMI')
ax.plot(df_valid['grid_size'], df_valid['FMI'], 'r-', linewidth=1.5, marker='^', markersize=2, label='FMI')
ax.set_xlabel('grid_size')
ax.set_ylabel('Score')
ax.grid(True, alpha=0.3)
ax.set_xlim([df_valid['grid_size'].min(), df_valid['grid_size'].max()])
ax.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stability_metrics_Jain.png'), dpi=300, bbox_inches='tight')
plt.close()
print('已保存: stability_metrics_Jain.png')

# ========== 图3: Silhouette ==========
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(df_valid['grid_size'], df_valid['Silhouette'], 'm-', linewidth=1.5, marker='o', markersize=2)
ax.set_xlabel('grid_size')
ax.set_ylabel('Silhouette')
ax.grid(True, alpha=0.3)
ax.set_xlim([df_valid['grid_size'].min(), df_valid['grid_size'].max()])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stability_Silhouette_Jain.png'), dpi=300, bbox_inches='tight')
plt.close()
print('已保存: stability_Silhouette_Jain.png')

# ========== 图4: CH ==========
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(df_valid['grid_size'], df_valid['CH'], 'c-', linewidth=1.5, marker='o', markersize=2)
ax.set_xlabel('grid_size')
ax.set_ylabel('CH Index')
ax.grid(True, alpha=0.3)
ax.set_xlim([df_valid['grid_size'].min(), df_valid['grid_size'].max()])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stability_CH_Jain.png'), dpi=300, bbox_inches='tight')
plt.close()
print('已保存: stability_CH_Jain.png')

# 打印最佳值
best_ari_idx = df_valid['ARI'].idxmax()
best_row = df_valid.loc[best_ari_idx]
print(f"\n最佳ARI: {best_row['ARI']:.4f} @ grid_size={best_row['grid_size']:.2f}")
print(f"对应簇数: {best_row['n_clusters']}")
