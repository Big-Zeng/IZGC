"""
DPC-FWSN 无监督聚类 (MATLAB)
使用轮廓系数和CH指数作为优化目标
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_comparison.utils import (
    load_dataset, normalize_coordinates,
    evaluate_clustering_unsupervised,
    plot_clustering_result
)

# 尝试导入MATLAB相关模块
try:
    from algorithm_comparison.alg_dpc_fwsn import run_dpc_fwsn, MATLAB_AVAILABLE
except:
    MATLAB_AVAILABLE = False

# ==================== 配置 ====================
DATASET = "../data/real_data/湘潭市_real.csv"
OUTPUT_DIR = "results_unsupervised"
NORMALIZE = True

# 优化目标: 'Silhouette' 或 'CH'
OPTIMIZE_METRIC = 'Silhouette'

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'k': (1, 50, 1)
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
    if not MATLAB_AVAILABLE:
        print("错误: MATLAB不可用，无法运行DPC-FWSN")
        print("请确保已安装MATLAB Engine for Python")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("DPC-FWSN 聚类算法 - 无监督版本 (MATLAB)")
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
    k_values = generate_param_values(*PARAM_RANGES['k'])
    print(f"  k: {PARAM_RANGES['k'][0]} -> {PARAM_RANGES['k'][1]}, 步长={PARAM_RANGES['k'][2]}, 共{len(k_values)}个值")
    
    print(f"参数组合数: {len(k_values)}")
    print(f"\n开始优化...")
    
    try:
        from matlab_integration import MatlabRunner
        matlab_runner = MatlabRunner(method='auto')
        
        best_score = -np.inf
        best_params = None
        best_labels = None
        best_metrics = None
        
        # 无监督模式：假设簇数为数据量的平方根
        n_clusters = max(2, int(np.sqrt(len(X) / 10)))
        # 创建假的标签用于MATLAB接口
        fake_labels = np.zeros(len(X))
        data_with_labels = np.column_stack([X, fake_labels])
        
        start_time = time.time()
        
        for i, k in enumerate(k_values, 1):
            try:
                labels, _ = matlab_runner.run_algorithm_pr(
                    data_with_labels, n_clusters, k
                )
                if labels is not None:
                    metrics = evaluate_clustering_unsupervised(labels, X)
                    if metrics['n_clusters'] < 2:
                        continue
                    score = metrics[OPTIMIZE_METRIC]
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'k': k}
                        best_labels = labels.copy()
                        best_metrics = metrics.copy()
                    
                    if i % 10 == 0 or i == len(k_values):
                        print(f"  进度: {i}/{len(k_values)}, 当前最佳{OPTIMIZE_METRIC}: {best_score:.4f}")
            except Exception as e:
                if i % 10 == 0:
                    print(f"  k={k}: 失败 - {str(e)[:50]}")
        
        matlab_runner.close()
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
            
            # 保存图表
            plot_clustering_result(
                X, best_labels,
                f"DPC-FWSN (Sil={best_metrics['Silhouette']:.3f})",
                f"{OUTPUT_DIR}/DPC-FWSN_{dataset_name}.png", dpi=300
            )
            
            # 保存最佳结果到txt
            txt_path = f"{OUTPUT_DIR}/DPC-FWSN_{dataset_name}_best.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"算法: DPC-FWSN\n")
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
        else:
            print(f"\n未找到有效参数")
            
    except Exception as e:
        print(f"DPC-FWSN失败: {e}")
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
