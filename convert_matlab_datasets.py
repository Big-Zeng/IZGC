"""
将MATLAB数据集转换为CSV格式
从 compare/dataset/shape/*.txt 转换到 data/*.csv
"""

import os
import pandas as pd
import glob

def convert_txt_to_csv(input_file, output_file):
    """
    将MATLAB的txt数据集转换为CSV格式
    
    参数:
        input_file: 输入的txt文件路径
        output_file: 输出的csv文件路径
    """
    try:
        # 读取txt文件（制表符分隔）
        data = pd.read_csv(input_file, sep='\t', header=None)
        
        # 检查列数
        if data.shape[1] == 3:
            # 三列：x, y, target
            data.columns = ['x', 'y', 'target']
        elif data.shape[1] > 3:
            # 多列：前n-1列是特征，最后一列是标签
            feature_cols = [f'feature_{i}' for i in range(data.shape[1] - 1)]
            data.columns = feature_cols + ['target']
        else:
            print(f"警告: {input_file} 列数不足3列，跳过")
            return False
        
        # 保存为CSV
        data.to_csv(output_file, index=False)
        print(f"✓ 转换成功: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        print(f"  样本数: {len(data)}, 类别数: {data['target'].nunique()}")
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {input_file}")
        print(f"  错误: {str(e)}")
        return False


def batch_convert_datasets():
    """批量转换数据集"""
    
    # 源目录和目标目录
    source_dir = "compare/dataset/shape"
    target_dir = "data"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(source_dir, "*.txt"))
    
    print("=" * 70)
    print("批量转换MATLAB数据集到CSV格式")
    print("=" * 70)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"找到 {len(txt_files)} 个txt文件\n")
    
    success_count = 0
    fail_count = 0
    
    for txt_file in txt_files:
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        
        # 输出文件路径
        csv_file = os.path.join(target_dir, f"{base_name}.csv")
        
        # 转换
        if convert_txt_to_csv(txt_file, csv_file):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 70)
    print(f"转换完成！成功: {success_count}, 失败: {fail_count}")
    print("=" * 70)
    
    # 列出转换后的文件
    print("\n转换后的CSV文件:")
    csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
    for i, csv_file in enumerate(csv_files, 1):
        file_size = os.path.getsize(csv_file) / 1024  # KB
        print(f"  {i:2d}. {os.path.basename(csv_file):<30} ({file_size:.1f} KB)")


def convert_single_dataset(dataset_name):
    """
    转换单个数据集
    
    参数:
        dataset_name: 数据集名称（不含扩展名），如 'flame', 'Aggregation'
    """
    source_file = f"compare/dataset/shape/{dataset_name}.txt"
    target_file = f"data/{dataset_name}.csv"
    
    if not os.path.exists(source_file):
        print(f"错误: 文件不存在 {source_file}")
        return False
    
    print(f"转换单个数据集: {dataset_name}")
    return convert_txt_to_csv(source_file, target_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 转换指定的数据集
        dataset_name = sys.argv[1]
        convert_single_dataset(dataset_name)
    else:
        # 批量转换所有数据集
        batch_convert_datasets()
