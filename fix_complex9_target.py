"""
将 complex9.csv 的 target 列 +1
"""
import pandas as pd

# 读取数据
df = pd.read_csv('data/complex8.csv')

print("修改前:")
print(f"  target 范围: {df['target'].min()} ~ {df['target'].max()}")
print(f"  唯一值: {sorted(df['target'].unique())}")

# target +1
df['target'] = df['target'] + 1

print("\n修改后:")
print(f"  target 范围: {df['target'].min()} ~ {df['target'].max()}")
print(f"  唯一值: {sorted(df['target'].unique())}")

# 保存
df.to_csv('data/complex8_1.csv', index=False)
print("\n✓ 已保存到 data/complex8_1.csv")
