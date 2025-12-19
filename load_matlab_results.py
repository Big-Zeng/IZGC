"""
从MATLAB保存的结果中读取数据
支持JSON和MAT两种格式
"""

import json
import pandas as pd
import numpy as np
import os

def load_matlab_results_from_json(json_file='compare/matlab_results/results.json'):
    """
    从JSON文件加载MATLAB结果
    
    参数:
        json_file: JSON文件路径
        
    返回:
        dict: 包含所有数据集结果的字典
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON文件不存在: {json_file}")
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    return results


def load_matlab_results_from_mat(mat_file='compare/matlab_results/all_results.mat'):
    """
    从MAT文件加载MATLAB结果
    
    参数:
        mat_file: MAT文件路径
        
    返回:
        dict: 包含所有数据集结果的字典
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("需要安装scipy: pip install scipy")
    
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"MAT文件不存在: {mat_file}")
    
    mat_data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    results = {}
    
    # 解析MAT文件结构
    all_results = mat_data['all_results']
    
    # 获取所有字段名（数据集名称）
    field_names = all_results._fieldnames
    
    for name in field_names:
        dataset_result = getattr(all_results, name)
        results[name] = {
            'AMI': float(dataset_result.AMI),
            'ARI': float(dataset_result.ARI),
            'FMI': float(dataset_result.FMI),
            'n_clusters': int(dataset_result.n_clusters),
            'n_samples': int(dataset_result.n_samples),
            'labels': np.array(dataset_result.labels).flatten()
        }
    
    return results


def load_matlab_labels(dataset_name, labels_dir='compare/matlab_results'):
    """
    加载特定数据集的MATLAB聚类标签
    
    参数:
        dataset_name: 数据集名称
        labels_dir: 标签文件目录
        
    返回:
        numpy.ndarray: 聚类标签数组
    """
    labels_file = os.path.join(labels_dir, f'{dataset_name}_labels.csv')
    
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"标签文件不存在: {labels_file}")
    
    labels = pd.read_csv(labels_file, header=None).values.flatten()
    return labels


def get_matlab_result(dataset_name, prefer_format='json'):
    """
    获取指定数据集的MATLAB结果
    
    参数:
        dataset_name: 数据集名称
        prefer_format: 优先格式 ('json' 或 'mat')
        
    返回:
        dict: 包含AMI、ARI、FMI等指标的字典
    """
    try:
        if prefer_format == 'json':
            results = load_matlab_results_from_json()
        else:
            results = load_matlab_results_from_mat()
        
        if dataset_name in results:
            return results[dataset_name]
        else:
            raise KeyError(f"数据集 '{dataset_name}' 不在结果中")
    
    except FileNotFoundError:
        # 尝试另一种格式
        try:
            if prefer_format == 'json':
                results = load_matlab_results_from_mat()
            else:
                results = load_matlab_results_from_json()
            
            if dataset_name in results:
                return results[dataset_name]
            else:
                raise KeyError(f"数据集 '{dataset_name}' 不在结果中")
        except FileNotFoundError:
            raise FileNotFoundError(
                "MATLAB结果文件未找到。请先在MATLAB中运行 run_and_save_results.m"
            )


def list_available_datasets():
    """
    列出所有可用的数据集
    
    返回:
        list: 数据集名称列表
    """
    try:
        results = load_matlab_results_from_json()
    except FileNotFoundError:
        try:
            results = load_matlab_results_from_mat()
        except FileNotFoundError:
            return []
    
    return list(results.keys())


def print_matlab_results_summary():
    """打印MATLAB结果汇总"""
    try:
        results = load_matlab_results_from_json()
    except FileNotFoundError:
        try:
            results = load_matlab_results_from_mat()
        except FileNotFoundError:
            print("错误: MATLAB结果文件未找到")
            print("请先在MATLAB中运行: run_and_save_results.m")
            return
    
    print("=" * 80)
    print("MATLAB Algorithm_PR 结果汇总")
    print("=" * 80)
    print(f"{'数据集':<15} {'AMI':<10} {'ARI':<10} {'FMI':<10} {'样本数':<10} {'簇数':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<15} {result['AMI']:<10.4f} {result['ARI']:<10.4f} "
              f"{result['FMI']:<10.4f} {result['n_samples']:<10} {result['n_clusters']:<10}")
    
    print("=" * 80)


# 使用示例
if __name__ == "__main__":
    print("MATLAB结果加载工具\n")
    
    # 列出可用数据集
    datasets = list_available_datasets()
    if datasets:
        print(f"找到 {len(datasets)} 个数据集:")
        for name in datasets:
            print(f"  - {name}")
        print()
    else:
        print("未找到MATLAB结果文件")
        print("请先在MATLAB中运行: compare/algorithm/run_and_save_results.m\n")
        exit(1)
    
    # 打印汇总
    print_matlab_results_summary()
    
    # 示例：加载单个数据集
    print("\n示例：加载flame数据集")
    print("-" * 40)
    try:
        result = get_matlab_result('flame')
        print(f"AMI: {result['AMI']:.4f}")
        print(f"ARI: {result['ARI']:.4f}")
        print(f"FMI: {result['FMI']:.4f}")
        print(f"样本数: {result['n_samples']}")
        print(f"簇数: {result['n_clusters']}")
        print(f"标签前10个: {result['labels'][:10]}")
    except (FileNotFoundError, KeyError) as e:
        print(f"错误: {e}")
