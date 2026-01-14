"""
批量更新所有无监督算法脚本
添加记录所有评估结果和前10名的功能
"""

import os
import re

# 要更新的文件列表
files_to_update = [
    'run_kmeans_unsupervised.py',
    'run_contour_unsupervised.py',
    'run_clique_unsupervised.py',
    'run_gmm_unsupervised.py',
    'run_bang_unsupervised.py',
]

base_dir = 'algorithm_comparison'

for filename in files_to_update:
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"跳过: {filename} (文件不存在)")
        continue
    
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有 all_results
    if 'all_results = []' in content:
        print(f"跳过: {filename} (已更新)")
        continue
    
    # 1. 添加 all_results 列表初始化
    # 在 best_metrics = None 后添加
    content = re.sub(
        r'(best_metrics = None)\n',
        r'\1\n    all_results = []  # 记录所有有效结果\n',
        content
    )
    
    # 2. 在记录best_score之前添加记录all_results的代码
    # 查找 if score > best_score: 之前，添加记录代码
    content = re.sub(
        r'(\s+)(score = metrics\[OPTIMIZE_METRIC\])\n(\s+if score > best_score:)',
        r'\1\2\n\n\1# 记录所有有效结果\n\1result_record = {\n\1    \'params\': params.copy(),\n\1    \'metrics\': metrics.copy(),\n\1    \'score\': score\n\1}\n\1all_results.append(result_record)\n\3',
        content
    )
    
    # 3. 更新进度显示，添加有效结果计数
    content = re.sub(
        r'print\(f"  进度: \{i\}/\{total\}, 当前最佳\{OPTIMIZE_METRIC\}: \{best_score:.4f\}"\)',
        r'print(f"  进度: {i}/{total}, 当前最佳{OPTIMIZE_METRIC}: {best_score:.4f}, 有效结果: {len(all_results)}")',
        content
    )
    
    # 4. 添加总评估次数显示
    content = re.sub(
        r'(print\(f"用时: \{elapsed:.2f\}s"\))\n',
        r'\1\n        print(f"总评估次数: {len(all_results)}")\n',
        content
    )
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已更新: {filename}")

print("\n批量更新完成！")
print("注意: 仍需手动添加保存all_results和top10的完整代码")
