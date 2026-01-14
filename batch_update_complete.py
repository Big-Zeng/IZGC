"""
为所有无监督脚本添加保存all_results和top10的完整代码
"""

import os
import re

# 要更新的文件和算法名称映射
files_config = {
    'run_kmeans_unsupervised.py': 'KMeans',
    'run_contour_unsupervised.py': 'ContourBased',
    'run_clique_unsupervised.py': 'CLIQUE',
    'run_gmm_unsupervised.py': 'GMM',
    'run_bang_unsupervised.py': 'BANG',
}

base_dir = 'algorithm_comparison'

# 生成保存all_results和top10的代码模板
def generate_save_code(alg_name):
    return f'''        
        # 保存所有评估结果
        if all_results:
            all_results_path = f"{{OUTPUT_DIR}}/{alg_name}_{{dataset_name}}_all_results.txt"
            with open(all_results_path, 'w', encoding='utf-8') as f:
                f.write(f"{alg_name} - 所有评估结果\\n")
                f.write(f"数据集: {{dataset_name}}\\n")
                f.write(f"优化目标: {{OPTIMIZE_METRIC}}\\n")
                f.write(f"总评估次数: {{len(all_results)}}\\n")
                f.write(f"{{'='*100}}\\n\\n")
                
                for idx, result in enumerate(all_results, 1):
                    f.write(f"评估 #{{idx}}\\n")
                    f.write(f"  参数: {{result['params']}}\\n")
                    m = result['metrics']
                    f.write(f"  Silhouette: {{m['Silhouette']:.4f}}\\n")
                    f.write(f"  CH: {{m['CH']:.2f}}\\n")
                    dbi_val = m.get('DBI', float('inf'))
                    if dbi_val == float('inf'):
                        f.write(f"  DBI: inf\\n")
                    else:
                        f.write(f"  DBI: {{dbi_val:.4f}}\\n")
                    f.write(f"  簇数: {{m['n_clusters']}}, 噪声: {{m['n_noise']}}\\n")
                    f.write(f"  {{OPTIMIZE_METRIC}}: {{result['score']:.4f}}\\n")
                    f.write(f"-" * 100 + "\\n")
            print(f"  已保存所有结果: {{all_results_path}}")
            
            # 排序并显示/保存前10名
            sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
            top10 = sorted_results[:10]
            
            print(f"\\n{{'='*70}}")
            print(f"排名前10的参数组合（按{{OPTIMIZE_METRIC}}排序）")
            print(f"{{'='*70}}")
            for rank, result in enumerate(top10, 1):
                m = result['metrics']
                print(f"\\n第 {{rank}} 名:")
                print(f"  参数: {{result['params']}}")
                print(f"  {{OPTIMIZE_METRIC}}: {{result['score']:.4f}}")
                print(f"  Silhouette: {{m['Silhouette']:.4f}}, CH: {{m['CH']:.2f}}", end="")
                dbi_val = m.get('DBI', float('inf'))
                if dbi_val != float('inf'):
                    print(f", DBI: {{dbi_val:.4f}}", end="")
                print(f", 簇数: {{m['n_clusters']}}")
            
            # 保存前10名到txt
            top10_path = f"{{OUTPUT_DIR}}/{alg_name}_{{dataset_name}}_top10.txt"
            with open(top10_path, 'w', encoding='utf-8') as f:
                f.write(f"{alg_name} - 排名前10的参数组合\\n")
                f.write(f"数据集: {{dataset_name}}\\n")
                f.write(f"优化目标: {{OPTIMIZE_METRIC}}\\n")
                f.write(f"{{'='*100}}\\n\\n")
                
                for rank, result in enumerate(top10, 1):
                    m = result['metrics']
                    f.write(f"第 {{rank}} 名\\n")
                    f.write(f"  参数: {{result['params']}}\\n")
                    f.write(f"  {{OPTIMIZE_METRIC}}: {{result['score']:.4f}}\\n")
                    f.write(f"  Silhouette: {{m['Silhouette']:.4f}}\\n")
                    f.write(f"  CH: {{m['CH']:.2f}}\\n")
                    dbi_val = m.get('DBI', float('inf'))
                    if dbi_val == float('inf'):
                        f.write(f"  DBI: inf\\n")
                    else:
                        f.write(f"  DBI: {{dbi_val:.4f}}\\n")
                    f.write(f"  簇数: {{m['n_clusters']}}, 噪声: {{m['n_noise']}}\\n")
                    f.write(f"-" * 100 + "\\n")
            print(f"\\n  已保存前10名结果: {{top10_path}}")'''

for filename, alg_name in files_config.items():
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"跳过: {filename} (文件不存在)")
        continue
    
    # 读取文件
    with open(filepath, 'w', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有 all_results_path
    if 'all_results_path' in content:
        print(f"跳过: {filename} (已添加完整保存代码)")
        continue
    
    # 在 print(f"  已保存: {txt_path}") 后添加新代码
    save_code = generate_save_code(alg_name)
    content = re.sub(
        r'(print\(f"  已保存: \{txt_path\}"\))\n',
        r'\1' + save_code + '\n',
        content
    )
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已更新: {filename}")

print("\n完整更新完成！")
