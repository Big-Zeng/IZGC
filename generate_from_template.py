"""
从DBSCAN模板生成其他算法的完整脚本
"""

import os
import re

# 读取DBSCAN模板
with open('algorithm_comparison/run_dbscan_unsupervised.py', 'r', encoding='utf-8') as f:
    template = f.read()

# 算法配置
algorithms = {
    'run_kmeans_unsupervised.py': {
        'name': 'KMeans',
        'display_name': 'K-means++',
        'import_func': 'run_kmeans',
        'import_module': 'alg_kmeans',
        'param_ranges': """{
    'n_clusters': (2, 20, 1),
    'n_init': (10, 10, 10),
    'max_iter': (300, 300, 100)
}""",
        'progress_mod': 20
    },
    'run_contour_unsupervised.py': {
        'name': 'ContourBased',
        'display_name': 'ContourBased',
        'import_func': 'run_contour_clustering',
        'import_module': 'alg_contour',
        'param_ranges': """{
    'grid_size': (1, 5, 0.1),
    'max_buffer_distance': (0, 0, 1),
    'min_cluster_size': (5, 5, 1),
    'density_radius': (30, 30, 1)
}""",
        'progress_mod': 50
    },
    'run_clique_unsupervised.py': {
        'name': 'CLIQUE',
        'display_name': 'CLIQUE',
        'import_func': 'run_clique2',
        'import_module': 'alg_clique2',
        'param_ranges': """{
    'intervals': (4, 100, 1),
    'density_threshold': (1, 50, 1)
}""",
        'progress_mod': 50
    },
    'run_gmm_unsupervised.py': {
        'name': 'GMM',
        'display_name': 'GMM (高斯混合模型)',
        'import_func': 'run_gmm',
        'import_module': 'alg_gmm',
        'param_ranges': """{
    'n_components': (2, 20, 1),
    'n_init': (10, 50, 10)
}""",
        'progress_mod': 20,
        'special': 'gmm'  # GMM需要特殊处理covariance_type
    },
    'run_bang_unsupervised.py': {
        'name': 'BANG',
        'display_name': 'BANG',
        'import_func': 'run_bang',
        'import_module': 'alg_bang',
        'param_ranges': """{
    'levels': (4, 50, 1),
    'density_threshold': (0.01, 1, 0.05)
}""",
        'progress_mod': 50,
        'additional_import': ', PYCLUSTERING_AVAILABLE'
    }
}

for filename, config in algorithms.items():
    print(f"\n生成 {filename}...")
    
    # 替换基本信息
    content = template.replace('DBSCAN', config['name'])
    content = content.replace('run_dbscan', config['import_func'])
    content = content.replace('from algorithm_comparison.alg_dbscan import run_dbscan',
                            f"from algorithm_comparison.{config['import_module']} import {config['import_func']}{config.get('additional_import', '')}")
    
    # 只在标题中使用display_name
    content = re.sub(
        r'print\("DBSCAN 聚类算法 - 无监督版本"\)',
        f'print("{config["display_name"]} 聚类算法 - 无监督版本")',
        content, count=1
    )
    
    # 替换参数范围（保留原有配置）
    # 我们不替换PARAM_RANGES，保持用户已设置的参数
    
    # 替换进度显示频率
    if 'progress_mod' in config:
        content = re.sub(
            r'if i % \d+ == 0 or i == total:',
            f'if i % {config["progress_mod"]} == 0 or i == total:',
            content
        )
    
    # 保存文件（备份原文件）
    output_path = f'algorithm_comparison/{filename}'
    if os.path.exists(output_path):
        backup_path = f'algorithm_comparison/{filename}.backup'
        os.rename(output_path, backup_path)
        print(f"  已备份原文件到: {backup_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ 已生成: {output_path}")

print("\n\n所有文件生成完成！")
print("注意：GMM算法需要特殊处理covariance_type，请手动检查run_gmm_unsupervised.py")
