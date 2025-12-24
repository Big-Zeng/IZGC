"""
快速测试批量测试功能（只测试1-2个数据集）
"""

import os
import shutil

# 临时修改配置
import batch_test_all_datasets as batch_module

# 只启用KMeans和Contour-Based（跳过DBSCAN和MATLAB以加快速度）
batch_module.ALGORITHMS_CONFIG['KMeans']['enabled'] = True
batch_module.ALGORITHMS_CONFIG['DBSCAN']['enabled'] = False
batch_module.ALGORITHMS_CONFIG['Contour-Based']['enabled'] = True
batch_module.ALGORITHMS_CONFIG['Contour-Based']['params']['optimize_grid'] = False
batch_module.ALGORITHMS_CONFIG['MATLAB-Algorithm_PR']['enabled'] = False

# 修改输出文件名
batch_module.OUTPUT_FILE = "test_results.txt"

print("="*80)
print("快速测试批量测试功能")
print("="*80)
print("配置:")
print("  - 只测试 KMeans 和 Contour-Based")
print("  - 跳过 DBSCAN 和 MATLAB")
print("  - 不优化 grid_size")
print("  - 只测试 data 文件夹中的数据集")
print("="*80)

# 创建测试器
tester = batch_module.BatchTester(batch_module.DATA_FOLDER, batch_module.OUTPUT_FOLDER)

# 获取数据集列表
datasets = tester.get_dataset_files()
print(f"\n找到 {len(datasets)} 个数据集")

if len(datasets) == 0:
    print("错误: data 文件夹中没有CSV文件")
    exit(1)

# 只测试前2个数据集
test_datasets = datasets[:min(2, len(datasets))]
print(f"将测试前 {len(test_datasets)} 个数据集: {[name for name, _ in test_datasets]}")

# 测试
all_results = []
for i, (filename, filepath) in enumerate(test_datasets, 1):
    print(f"\n[{i}/{len(test_datasets)}]", end=' ')
    result = tester.test_dataset(filename, filepath)
    all_results.append(result)

# 保存结果
print(f"\n{'='*80}")
print("保存结果...")
tester.save_results(all_results, batch_module.OUTPUT_FILE)

print(f"{'='*80}")
print("快速测试完成！")
print(f"{'='*80}")
print(f"\n结果保存在: {os.path.join(batch_module.OUTPUT_FOLDER, batch_module.OUTPUT_FILE)}")
print("\n如果测试成功，可以运行完整版本:")
print("  python batch_test_all_datasets.py")
