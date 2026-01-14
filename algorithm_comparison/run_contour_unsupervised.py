"""
ContourBased 无监督聚类
使用轮廓系数和CH指数作为优化目标
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import time
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates,
    evaluate_clustering_unsupervised,
    plot_clustering_result
)
from algorithm_comparison.alg_contour import run_contour_clustering

# ==================== 配置 ====================
DATASET = "../data/real_data/长沙市_real.csv"
OUTPUT_DIR = "results_unsupervised"
NORMALIZE = True

# 优化目标: 'Silhouette' 或 'CH'
OPTIMIZE_METRIC = 'Silhouette'

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'grid_size': (0.5, 5, 0.001),
    'max_buffer_distance': (0, 0, 1),
    'min_cluster_size': (5, 5, 1),
    'density_radius': (0.01, 0.01, 0.001)
}
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("ContourBased 聚类算法 - 无监督版本")
    print(f"优化目标: {OPTIMIZE_METRIC}")
    print("="*70)
    
    # 加载数据
    X, y_true = load_dataset(DATASET)
    dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
    
    print(f"\n数据集: {dataset_name}")
    print(f"样本数: {len(X)}")
    if y_true is not None:
        print(f"真实簇数: {len(np.unique(y_true))}")
    else:
        print("无真实标签（无监督模式）")
    
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")
    
    # 生成参数值
    param_grid = {}
    for pname, (start, end, step) in PARAM_RANGES.items():
        param_grid[pname] = generate_param_values(start, end, step)
        print(f"  {pname}: {start} -> {end}, 步长={step}, 共{len(param_grid[pname])}个值")
    
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_metrics = None
    all_results = []  # 记录所有有效结果
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total = int(np.prod([len(v) for v in param_values]))
    
    print(f"参数组合数: {total}")
    print(f"\n开始优化...")
    
    start_time = time.time()
    
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        try:
            labels = run_contour_clustering(X, **params)
            if labels is None:
                continue
            
            metrics = evaluate_clustering_unsupervised(labels, X)
            
            if metrics['n_clusters'] < 2:
                continue
            
            score = metrics[OPTIMIZE_METRIC]
            
            # 记录所有有效结果
            result_record = {
                'params': params.copy(),
                'metrics': metrics.copy(),
                'score': score
            }
            all_results.append(result_record)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_labels = labels.copy() if isinstance(labels, np.ndarray) else np.array(labels)
                best_metrics = metrics.copy()
                
            if i % 50 == 0 or i == total:
                print(f"  进度: {i}/{total}, 当前最佳{OPTIMIZE_METRIC}: {best_score:.4f}, 有效结果: {len(all_results)}")
                
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    
    if best_params:
        print(f"\n{'='*70}")
        print("最佳结果")
        print(f"{'='*70}")
        print(f"最佳参数: {best_params}")
        print(f"最佳{OPTIMIZE_METRIC}: {best_score:.4f}")
        print(f"Silhouette: {best_metrics['Silhouette']:.4f}")
        print(f"CH: {best_metrics['CH']:.2f}")
        if 'DBI' in best_metrics:
            dbi_val = best_metrics['DBI']
            if dbi_val == float('inf'):
                print(f"DBI: inf")
            else:
                print(f"DBI: {dbi_val:.4f}")
        print(f"簇数: {best_metrics['n_clusters']}, 噪声: {best_metrics['n_noise']}")
        print(f"用时: {elapsed:.2f}s")
        print(f"总评估次数: {len(all_results)}")
        
        # 保存图表
        plot_clustering_result(
            X, best_labels,
            f"ContourBased (Sil={best_metrics['Silhouette']:.3f})",
            f"{OUTPUT_DIR}/ContourBased_{dataset_name}.png", dpi=300
        )
        
        # 保存最佳结果到txt
        txt_path = f"{OUTPUT_DIR}/ContourBased_{dataset_name}_best.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"算法: ContourBased\n")
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"最佳参数: {best_params}\n")
            f.write(f"\n评估指标（无监督）:\n")
            f.write(f"  Silhouette: {best_metrics['Silhouette']:.4f}\n")
            f.write(f"  CH: {best_metrics['CH']:.2f}\n")
            if 'DBI' in best_metrics:
                dbi_val = best_metrics['DBI']
                if dbi_val == float('inf'):
                    f.write(f"  DBI: inf\n")
                else:
                    f.write(f"  DBI: {dbi_val:.4f}\n")
            f.write(f"  簇数: {best_metrics['n_clusters']}\n")
            f.write(f"  噪声点: {best_metrics['n_noise']}\n")
        print(f"  已保存: {txt_path}")
        
        # 保存所有评估结果
        if all_results:
            all_results_path = f"{OUTPUT_DIR}/ContourBased_{dataset_name}_all_results.txt"
            with open(all_results_path, 'w', encoding='utf-8') as f:
                f.write(f"ContourBased - 所有评估结果\n")
                f.write(f"数据集: {dataset_name}\n")
                f.write(f"优化目标: {OPTIMIZE_METRIC}\n")
                f.write(f"总评估次数: {len(all_results)}\n")
                f.write(f"{'='*100}\n\n")
                
                for idx, result in enumerate(all_results, 1):
                    f.write(f"评估 #{idx}\n")
                    f.write(f"  参数: {result['params']}\n")
                    m = result['metrics']
                    f.write(f"  Silhouette: {m['Silhouette']:.4f}\n")
                    f.write(f"  CH: {m['CH']:.2f}\n")
                    dbi_val = m.get('DBI', float('inf'))
                    if dbi_val == float('inf'):
                        f.write(f"  DBI: inf\n")
                    else:
                        f.write(f"  DBI: {dbi_val:.4f}\n")
                    f.write(f"  簇数: {m['n_clusters']}, 噪声: {m['n_noise']}\n")
                    f.write(f"  {OPTIMIZE_METRIC}: {result['score']:.4f}\n")
                    f.write(f"-" * 100 + "\n")
            print(f"  已保存所有结果: {all_results_path}")
            
            # 排序并显示/保存前10名
            sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
            top10 = sorted_results[:10]
            
            print(f"\n{'='*70}")
            print(f"排名前10的参数组合（按{OPTIMIZE_METRIC}排序）")
            print(f"{'='*70}")
            for rank, result in enumerate(top10, 1):
                m = result['metrics']
                print(f"\n第 {rank} 名:")
                print(f"  参数: {result['params']}")
                print(f"  {OPTIMIZE_METRIC}: {result['score']:.4f}")
                print(f"  Silhouette: {m['Silhouette']:.4f}, CH: {m['CH']:.2f}", end="")
                dbi_val = m.get('DBI', float('inf'))
                if dbi_val != float('inf'):
                    print(f", DBI: {dbi_val:.4f}", end="")
                print(f", 簇数: {m['n_clusters']}")
            
            # 保存前10名到txt
            top10_path = f"{OUTPUT_DIR}/ContourBased_{dataset_name}_top10.txt"
            with open(top10_path, 'w', encoding='utf-8') as f:
                f.write(f"ContourBased - 排名前10的参数组合\n")
                f.write(f"数据集: {dataset_name}\n")
                f.write(f"优化目标: {OPTIMIZE_METRIC}\n")
                f.write(f"{'='*100}\n\n")
                
                for rank, result in enumerate(top10, 1):
                    m = result['metrics']
                    f.write(f"第 {rank} 名\n")
                    f.write(f"  参数: {result['params']}\n")
                    f.write(f"  {OPTIMIZE_METRIC}: {result['score']:.4f}\n")
                    f.write(f"  Silhouette: {m['Silhouette']:.4f}\n")
                    f.write(f"  CH: {m['CH']:.2f}\n")
                    dbi_val = m.get('DBI', float('inf'))
                    if dbi_val == float('inf'):
                        f.write(f"  DBI: inf\n")
                    else:
                        f.write(f"  DBI: {dbi_val:.4f}\n")
                    f.write(f"  簇数: {m['n_clusters']}, 噪声: {m['n_noise']}\n")
                    f.write(f"-" * 100 + "\n")
            print(f"\n  已保存前10名结果: {top10_path}")
    else:
        print(f"\n未找到有效参数")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
