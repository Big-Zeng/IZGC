"""
合并长沙市指定区域的数据
保留：天心区、岳麓区、开福区、雨花区、望城区、芙蓉区
"""

import pandas as pd
import os

# 数据目录
data_dir = "data/real_data"

# 要保留的区域
districts = [
    "天心区",
    "岳麓区",
    "开福区",
    "雨花区",
    "望城区",
    "芙蓉区"
]

print("开始合并长沙市指定区域数据...")
print(f"保留区域: {', '.join(districts)}\n")

# 存储所有数据
all_data = []

# 读取每个区的数据
for district in districts:
    file_path = os.path.join(data_dir, f"长沙市_{district}_real.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        count = len(df)
        all_data.append(df)
        print(f"✓ {district}: {count} 个点")
    else:
        print(f"✗ {district}: 文件不存在")

# 合并所有数据
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 保存到新文件
    output_path = os.path.join(data_dir, "长沙市_real.csv")
    
    # 备份原文件
    if os.path.exists(output_path):
        backup_path = os.path.join(data_dir, "长沙市_real_backup.csv")
        os.rename(output_path, backup_path)
        print(f"\n原文件已备份至: {backup_path}")
    
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n合并完成！")
    print(f"总数据点: {len(merged_df)}")
    print(f"保存至: {output_path}")
else:
    print("\n错误: 没有找到任何有效的区域数据")
