"""
将 client_data_by_city 中的真实数据转换为算法可接受的格式
只转换城市级别数据（不分区）
"""

import pandas as pd
import os

# 源目录和目标目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "..", "map_visualization", "client_data_by_city")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "real_data")

# 只转换这些城市
TARGET_CITIES = ['长沙市', '株洲市', '湘潭市', '衡阳市', '益阳市', '常德市']


def convert_file(input_path, output_path):
    """转换单个文件"""
    df = pd.read_csv(input_path)
    
    # 提取经纬度，重命名为 x, y
    df_out = pd.DataFrame({
        'x': df['lon'],
        'y': df['lat']
    })
    
    df_out.to_csv(output_path, index=False)
    return len(df_out)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"源目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标城市: {TARGET_CITIES}")
    print("="*60)
    
    converted = []
    for city in TARGET_CITIES:
        input_path = os.path.join(SOURCE_DIR, f"{city}.csv")
        
        if not os.path.exists(input_path):
            print(f"✗ {city}.csv 不存在")
            continue
        
        output_name = f"{city}_real.csv"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        n_samples = convert_file(input_path, output_path)
        converted.append({'name': city, 'samples': n_samples, 'output': output_name})
        print(f"✓ {city}.csv -> {output_name} ({n_samples} 样本)")
    
    print("="*60)
    print(f"共转换 {len(converted)} 个城市数据")


if __name__ == "__main__":
    main()
