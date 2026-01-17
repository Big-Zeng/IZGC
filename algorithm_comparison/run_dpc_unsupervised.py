"""
DPC-FWSN 无监督聚类 (MATLAB)
使用复合评估指标作为优化目标
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
    plot_clustering_result,
    print_best_result,
    save_best_result
)

# ==================== 配置 ====================
DATASET = "../data/real_data/衡阳市_real.csv"
OUTPUT_DIR = "results_unsupervised"
NORMALIZE = True

# 优化目标: 'Composite', 'Silhouette', 'CH'
OPTIMIZE_METRIC = 'Composite'

# 参数搜索范围 (start, end, step)
PARAM_RANGES = {
    'n_clusters': (2, 15, 1),  # 簇数量
    'k': (1, 50, 2)            # 近邻数量
}
# ==============================================


def generate_param_values(start, end, step):
    """生成参数值列表"""
    if isinstance(start, int) and isinstance(step, int):
        return list(range(start, end + 1, step))
    else:
        return list(np.arange(start, end + step/2, step))


def run_dpc_unsupervised(X, n_clusters, k, matlab_runner):
    """
    运行DPC-FWSN无监督聚类
    
    参数:
        X: 坐标数组 (n_samples, 2)
        n_clusters: 簇数量
        k: 近邻数量
        matlab_runner: MatlabRunner实例
        
    返回:
        labels: 聚类标签
    """
    # 调用无监督版本的MATLAB函数（仅传入坐标）
    labels = matlab_runner.run_algorithm_pr_unsupervised(X, n_clusters, k)
    return labels


def main():
    """主函数"""
    # 尝试导入MATLAB集成模块
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from matlab_integration import MatlabRunner
        MATLAB_AVAILABLE = True
    except ImportError as e:
        print(f"错误: 无法导入MATLAB集成模块 - {e}")
        MATLAB_AVAILABLE = False
    
    if not MATLAB_AVAILABLE:
        print("错误: MATLAB集成模块不可用，无法运行DPC-FWSN")
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
        print(f"真实簇数: {len(np.unique(y_true))} (仅供参考，不参与优化)")
    else:
        print("无真实标签（无监督模式）")
    
    if NORMALIZE:
        X = normalize_coordinates(X)
        print("✓ 坐标已归一化到[0, 100]")
    
    # 生成参数值
    n_clusters_values = generate_param_values(*PARAM_RANGES['n_clusters'])
    k_values = generate_param_values(*PARAM_RANGES['k'])
    
    print(f"\n参数搜索范围:")
    print(f"  n_clusters: {PARAM_RANGES['n_clusters'][0]} -> {PARAM_RANGES['n_clusters'][1]}, 步长={PARAM_RANGES['n_clusters'][2]}, 共{len(n_clusters_values)}个值")
    print(f"  k: {PARAM_RANGES['k'][0]} -> {PARAM_RANGES['k'][1]}, 步长={PARAM_RANGES['k'][2]}, 共{len(k_values)}个值")
    
    total_combinations = len(n_clusters_values) * len(k_values)
    print(f"参数组合数: {total_combinations}")
    print(f"\n开始优化...")
    
    try:
        # 初始化MATLAB运行器
        matlab_runner = MatlabRunner(method='auto')
        print(f"MATLAB方法: {matlab_runner.method}")
        
        best_score = -np.inf
        best_params = None
        best_labels = None
        best_metrics = None
        
        start_time = time.time()
        iteration = 0
        
        for n_clusters in n_clusters_values:
            for k in k_values:
                iteration += 1
                
                try:
                    # 运行无监督DPC-FWSN
                    labels = run_dpc_unsupervised(X, n_clusters, k, matlab_runner)
                    
                    if labels is not None:
                        # 使用Python计算无监督评估指标
                        metrics = evaluate_clustering_unsupervised(labels, X)
                        
                        if metrics['n_clusters'] < 2:
                            continue
                        
                        score = metrics[OPTIMIZE_METRIC]
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'n_clusters': n_clusters, 'k': k}
                            best_labels = labels.copy()
                            best_metrics = metrics.copy()
                        
                        if iteration % 20 == 0 or iteration == total_combinations:
                            elapsed = time.time() - start_time
                            print(f"  进度: {iteration}/{total_combinations} ({100*iteration/total_combinations:.1f}%), "
                                  f"当前最佳{OPTIMIZE_METRIC}: {best_score:.4f}, 用时: {elapsed:.1f}s")
                            
                except Exception as e:
                    if iteration % 20 == 0:
                        print(f"  n_clusters={n_clusters}, k={k}: 失败 - {str(e)[:50]}")
        
        # 关闭MATLAB运行器
        matlab_runner.close()
        elapsed = time.time() - start_time
        
        if best_params:
            # 打印最佳结果
            print_best_result("DPC-FWSN", best_params, best_metrics)
            print(f"总用时: {elapsed:.2f}s")
            
            # 保存图表
            plot_clustering_result(
                X, best_labels,
                f"DPC-FWSN (Composite={best_metrics['Composite']:.3f})",
                f"{OUTPUT_DIR}/DPC-FWSN_{dataset_name}.png", dpi=300
            )
            
            # 保存最佳结果到txt
            save_best_result(
                "DPC-FWSN", dataset_name, best_params, best_metrics,
                f"{OUTPUT_DIR}/DPC-FWSN_{dataset_name}_best.txt"
            )
        else:
            print(f"\n未找到有效参数组合")
            
    except Exception as e:
        import traceback
        print(f"DPC-FWSN失败: {e}")
        traceback.print_exc()
    
    print(f"\n完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
