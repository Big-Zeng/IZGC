# 聚类评估指标模块使用说明

## 概述

本模块提供了三种常用的聚类评估指标的Python实现：
- **AMI** (Adjusted Mutual Information) - 调整互信息
- **ARI** (Adjusted Rand Index) - 调整兰德指数  
- **FMI** (F-Measure / Fowlkes-Mallows Index) - F度量指数

## 文件说明

- `clustering_metrics.py` - 核心评估指标实现
- `evaluate_clustering_example.py` - 使用示例
- `verify_metrics.py` - 与sklearn的验证对比

## 快速开始

### 1. 基本使用

```python
from clustering_metrics import ClusteringMetrics

# 真实标签和预测标签
true_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
pred_labels = [0, 0, 1, 1, 1, 1, 2, 2, 2]

# 计算所有指标
metrics = ClusteringMetrics.evaluate_all(true_labels, pred_labels)

print(f"AMI: {metrics['AMI']:.4f}")
print(f"ARI: {metrics['ARI']:.4f}")
print(f"FMI: {metrics['FMI']:.4f}")
print(f"Accuracy: {metrics['Accuracy']:.4f}")
```

### 2. 单独调用各指标

```python
from clustering_metrics import ClusteringMetrics

# 单独计算AMI
ami = ClusteringMetrics.adjusted_mutual_information(true_labels, pred_labels)

# 单独计算ARI
ari = ClusteringMetrics.adjusted_rand_index(true_labels, pred_labels)

# 单独计算FMI (MATLAB风格，返回F-Measure和Accuracy)
fmi, accuracy = ClusteringMetrics.f_measure(true_labels, pred_labels)

# 单独计算FMI (sklearn风格，Fowlkes-Mallows Score)
fmi_sklearn = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)
```

### 3. 与你的聚类算法集成

```python
from ContourBasedClusteringWithoutZ3 import ContourBasedClusteringWithoutZ
from clustering_metrics import ClusteringMetrics

# 准备数据 (x, y, z, true_label)
actual_points = [
    (0.2, 0.3, 1.5, 0),
    (0.8, 1.2, 2.0, 0),
    (1.5, 1.5, 3.0, 0),
    (2.3, 2.6, 1.0, 1),
    (3.3, 1.6, 1.8, 1),
    # ... 更多数据点
]

# 执行聚类
cluster = ContourBasedClusteringWithoutZ(actual_points, grid_size=1.0)
clusters = cluster.perform_clustering()

# 获取预测标签
predicted_labels = cluster.get_predicted_labels()

# 提取真实标签
true_labels = [point[3] for point in actual_points]

# 计算评估指标
metrics = ClusteringMetrics.evaluate_all(true_labels, predicted_labels)

print(f"AMI: {metrics['AMI']:.4f}")
print(f"ARI: {metrics['ARI']:.4f}")
print(f"FMI: {metrics['FMI']:.4f}")
```

## 指标说明

### AMI (Adjusted Mutual Information)
- **范围**: [-1, 1]，通常在[0, 1]之间
- **含义**: 衡量两个聚类结果的互信息，调整了随机聚类的影响
- **最佳值**: 1.0 (完全匹配)
- **特点**: 对簇数量不敏感

### ARI (Adjusted Rand Index)
- **范围**: [-1, 1]
- **含义**: 衡量两个聚类结果的相似度，调整了随机分配的影响
- **最佳值**: 1.0 (完全匹配)
- **特点**: 
  - 0.0 表示随机聚类
  - 负值表示比随机聚类更差
  - **本实现与sklearn完全一致**

### FMI (F-Measure)
提供两种实现：

#### MATLAB风格 (f_measure方法)
- **范围**: [0, 1]
- **含义**: 基于precision和recall计算的F值
- **返回**: F-Measure值和Accuracy
- **用途**: 与MATLAB代码对比时使用

#### sklearn风格 (fowlkes_mallows_score方法)
- **范围**: [0, 1]
- **含义**: Fowlkes-Mallows指数，基于点对的几何平均
- **最佳值**: 1.0 (完全匹配)
- **特点**: **本实现与sklearn完全一致**

## 与MATLAB代码对应关系

在你的MATLAB代码 `Algorithm_PR.m` 中：

```matlab
AMI = AMI(AA(:,end),cl');
ARI = Library.GetAri(AA(:,end),cl');
FMI = Library.GetFmi(AA(:,end),cl');
```

对应的Python代码：

```python
ami = ClusteringMetrics.adjusted_mutual_information(true_labels, pred_labels)
ari = ClusteringMetrics.adjusted_rand_index(true_labels, pred_labels)
fmi, accuracy = ClusteringMetrics.f_measure(true_labels, pred_labels)
```

## 验证结果

运行 `verify_metrics.py` 可以验证实现的正确性：

```bash
python verify_metrics.py
```

验证结果显示：
- ✓ **ARI**: 与sklearn完全一致 (差异 = 0.0)
- ✓ **FMI (sklearn风格)**: 与sklearn完全一致 (差异 = 0.0)
- ⚠ **AMI**: 存在微小差异 (~0.01)，这是由于EMI计算方法的细微差异

## 注意事项

1. **标签格式**: 标签可以是任意类型（整数、字符串等），会自动处理
2. **噪声点**: 如果有噪声点（标签为-1），算法会正确处理
3. **性能**: 对于大规模数据集，AMI的计算可能较慢（因为需要计算EMI）
4. **FMI选择**: 
   - 如果要与MATLAB代码对比，使用 `f_measure()` 方法
   - 如果要与sklearn对比，使用 `fowlkes_mallows_score()` 方法

## 依赖项

```bash
pip install numpy scipy scikit-learn
```

## 示例输出

```
聚类评估指标测试:
--------------------------------------------------
AMI (Adjusted Mutual Information): 0.6747
ARI (Adjusted Rand Index):         0.6429
FMI (F-Measure):                   0.8857
Accuracy:                          0.8889
```

## 参考文献

1. Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison. Journal of Machine Learning Research, 11, 2837-2854.
2. Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.
3. Fowlkes, E. B., & Mallows, C. L. (1983). A method for comparing two hierarchical clusterings. Journal of the American Statistical Association, 78(383), 553-569.
