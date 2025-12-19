# 为MATLAB R2022配置Python 3.10环境

## MATLAB与Python版本兼容性

| MATLAB版本 | 支持的Python版本 |
|-----------|-----------------|
| R2022a | 3.8, 3.9 |
| R2022b | 3.8, 3.9, 3.10 |
| R2023a | 3.9, 3.10, 3.11 |
| R2023b | 3.9, 3.10, 3.11 |

**你的情况**: MATLAB R2022 → 需要Python 3.8-3.10

## 方案1: 使用虚拟环境（推荐）

这样可以保留你现有的Python环境，同时创建一个专门用于MATLAB的Python 3.10环境。

### 步骤1: 安装Python 3.10

**Windows**:
```bash
# 下载Python 3.10.11（稳定版）
# https://www.python.org/downloads/release/python-31011/
# 选择 "Windows installer (64-bit)"

# 安装时注意：
# ✓ 勾选 "Add Python 3.10 to PATH"
# ✓ 选择 "Customize installation"
# ✓ 安装到自定义目录，如: C:\Python310
```

### 步骤2: 创建虚拟环境

```bash
# 使用Python 3.10创建虚拟环境
C:\Python310\python.exe -m venv matlab_env

# 激活虚拟环境
# Windows CMD:
matlab_env\Scripts\activate.bat

# Windows PowerShell:
matlab_env\Scripts\Activate.ps1

# 验证Python版本
python --version
# 应该显示: Python 3.10.x
```

### 步骤3: 安装依赖包

```bash
# 在激活的虚拟环境中安装
pip install numpy pandas scipy scikit-learn matplotlib shapely

# 验证安装
python -c "import numpy; print(numpy.__version__)"
```

### 步骤4: 安装MATLAB Engine

```bash
# 找到MATLAB的Python引擎目录
# R2022a: C:\Program Files\MATLAB\R2022a\extern\engines\python
# R2022b: C:\Program Files\MATLAB\R2022b\extern\engines\python

# 进入该目录
cd "C:\Program Files\MATLAB\R2022a\extern\engines\python"

# 在虚拟环境中安装
python setup.py install

# 验证安装
python -c "import matlab.engine; print('MATLAB Engine安装成功')"
```

### 步骤5: 配置MATLAB使用Python 3.10

在MATLAB中运行：
```matlab
% 查看当前Python版本
pyenv

% 设置Python 3.10路径
pyenv('Version', 'C:\path\to\matlab_env\Scripts\python.exe')

% 验证
pyenv
```

### 步骤6: 在虚拟环境中运行你的脚本

```bash
# 激活虚拟环境
matlab_env\Scripts\activate.bat

# 运行对比测试
python compareTest_with_matlab.py
```

## 方案2: 使用Conda（更简单）

如果你已经安装了Anaconda或Miniconda：

```bash
# 创建Python 3.10环境
conda create -n matlab_env python=3.10

# 激活环境
conda activate matlab_env

# 安装依赖
conda install numpy pandas scipy scikit-learn matplotlib
pip install shapely

# 安装MATLAB Engine
cd "C:\Program Files\MATLAB\R2022a\extern\engines\python"
python setup.py install

# 运行脚本
python compareTest_with_matlab.py
```

## 方案3: 使用Octave（无需Python 3.10）

如果配置Python 3.10太麻烦，可以使用Octave（免费的MATLAB替代品）：

```bash
# 1. 安装Octave
# 下载: https://www.gnu.org/software/octave/download.html

# 2. 安装Oct2Py（在你当前的Python环境中）
pip install oct2py

# 3. 直接运行（不需要Python 3.10）
python compareTest_with_matlab.py
```

## 方案4: 最简单的方法 - 分离运行

如果上述方法都太复杂，可以：

### 步骤1: 在MATLAB中运行并保存结果

创建一个MATLAB脚本 `run_and_save_results.m`:

```matlab
% run_and_save_results.m
% 批量运行数据集并保存结果

datasets = {'flame', 'Aggregation', 'jain', 'spiral'};
results = struct();

for i = 1:length(datasets)
    dataset_name = datasets{i};
    fprintf('处理数据集: %s\n', dataset_name);
    
    % 读取数据
    data_file = sprintf('data/%s.csv', dataset_name);
    data = readtable(data_file);
    
    % 转换为矩阵
    AA = [data.x, data.y, data.target];
    
    % 运行Algorithm_PR
    KK = length(unique(AA(:,end)));  % 簇数
    k = 11;  % 近邻数
    
    [labels, ami, ari, fmi] = run_algorithm_pr_wrapper(AA, KK, k);
    
    % 保存结果
    results.(dataset_name).labels = labels;
    results.(dataset_name).AMI = ami;
    results.(dataset_name).ARI = ari;
    results.(dataset_name).FMI = fmi;
    
    % 保存标签到CSV
    csvwrite(sprintf('matlab_results/%s_labels.csv', dataset_name), labels);
end

% 保存所有结果
save('matlab_results/all_results.mat', 'results');

% 导出为JSON（可选）
jsonStr = jsonencode(results);
fid = fopen('matlab_results/results.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('所有结果已保存到 matlab_results/ 目录\n');
```

### 步骤2: 在Python中读取结果

创建 `load_matlab_results.py`:

```python
"""
从MATLAB保存的结果中读取数据
"""

import json
import pandas as pd
import numpy as np
from scipy.io import loadmat

def load_matlab_results_from_json(json_file='matlab_results/results.json'):
    """从JSON文件加载MATLAB结果"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results

def load_matlab_results_from_mat(mat_file='matlab_results/all_results.mat'):
    """从MAT文件加载MATLAB结果"""
    mat_data = loadmat(mat_file)
    results = {}
    
    # 解析MAT文件结构
    for key in mat_data.keys():
        if not key.startswith('__'):
            results[key] = {
                'AMI': float(mat_data[key]['AMI'][0][0]),
                'ARI': float(mat_data[key]['ARI'][0][0]),
                'FMI': float(mat_data[key]['FMI'][0][0]),
                'labels': mat_data[key]['labels'][0].flatten()
            }
    
    return results

def load_matlab_labels(dataset_name):
    """加载特定数据集的MATLAB聚类标签"""
    labels_file = f'matlab_results/{dataset_name}_labels.csv'
    labels = pd.read_csv(labels_file, header=None).values.flatten()
    return labels

# 使用示例
if __name__ == "__main__":
    # 方法1: 从JSON加载
    try:
        results = load_matlab_results_from_json()
        print("从JSON加载成功:")
        for dataset, metrics in results.items():
            print(f"\n{dataset}:")
            print(f"  AMI: {metrics['AMI']:.4f}")
            print(f"  ARI: {metrics['ARI']:.4f}")
            print(f"  FMI: {metrics['FMI']:.4f}")
    except FileNotFoundError:
        print("JSON文件未找到")
    
    # 方法2: 从MAT加载
    try:
        results = load_matlab_results_from_mat()
        print("\n从MAT文件加载成功")
    except FileNotFoundError:
        print("MAT文件未找到")
    
    # 方法3: 加载单个数据集的标签
    try:
        labels = load_matlab_labels('flame')
        print(f"\nflame数据集标签: {labels[:10]}...")
    except FileNotFoundError:
        print("标签文件未找到")
```

### 步骤3: 更新对比测试脚本

创建 `compareTest_with_saved_matlab.py`:

```python
"""
使用保存的MATLAB结果进行对比
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from clustering_metrics import ClusteringMetrics
from load_matlab_results import load_matlab_results_from_json, load_matlab_labels
import matplotlib.pyplot as plt

# 配置
DATASET = "flame"
N_CLUSTERS = 2

# 读取数据
df = pd.read_csv(f"data/{DATASET}.csv")
X = df[['x', 'y']].values
y_true = df['target'].values

print(f"数据集: {DATASET}")
print(f"样本数: {len(X)}, 真实簇数: {N_CLUSTERS}")

# 运行Python算法
print("\n运行Python算法...")

# KMeans
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_metrics = ClusteringMetrics.evaluate_all(y_true, kmeans_labels)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
dbscan_metrics = ClusteringMetrics.evaluate_all(y_true, dbscan_labels)

# 加载MATLAB结果
print("\n加载MATLAB结果...")
try:
    matlab_labels = load_matlab_labels(DATASET)
    matlab_metrics = ClusteringMetrics.evaluate_all(y_true, matlab_labels)
    
    # 或者直接从JSON加载指标
    # all_results = load_matlab_results_from_json()
    # matlab_metrics = all_results[DATASET]
    
    matlab_available = True
except FileNotFoundError:
    print("MATLAB结果未找到，请先在MATLAB中运行 run_and_save_results.m")
    matlab_available = False

# 打印对比
print("\n" + "=" * 80)
print("算法对比")
print("=" * 80)
print(f"{'算法':<20} {'AMI':<10} {'ARI':<10} {'FMI':<10}")
print("-" * 80)
print(f"{'KMeans':<20} {kmeans_metrics['AMI']:<10.4f} "
      f"{kmeans_metrics['ARI']:<10.4f} {kmeans_metrics['FMI']:<10.4f}")
print(f"{'DBSCAN':<20} {dbscan_metrics['AMI']:<10.4f} "
      f"{dbscan_metrics['ARI']:<10.4f} {dbscan_metrics['FMI']:<10.4f}")

if matlab_available:
    print(f"{'MATLAB Algorithm_PR':<20} {matlab_metrics['AMI']:<10.4f} "
          f"{matlab_metrics['ARI']:<10.4f} {matlab_metrics['FMI']:<10.4f}")

print("=" * 80)

# 可视化
if matlab_available:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=20)
    axes[0].set_title('Ground Truth')
    
    axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=20)
    axes[1].set_title(f'KMeans\nARI={kmeans_metrics["ARI"]:.3f}')
    
    axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='tab10', s=20)
    axes[2].set_title(f'DBSCAN\nARI={dbscan_metrics["ARI"]:.3f}')
    
    axes[3].scatter(X[:, 0], X[:, 1], c=matlab_labels, cmap='tab10', s=20)
    axes[3].set_title(f'MATLAB\nARI={matlab_metrics["ARI"]:.3f}')
    
    plt.tight_layout()
    plt.show()
```

## 推荐方案

根据你的情况，我推荐：

### 🥇 首选：方案4（分离运行）
**原因**:
- ✅ 最简单，不需要配置Python 3.10
- ✅ 不影响现有Python环境
- ✅ 一次运行，多次使用
- ✅ 结果可以保存和分享

**步骤**:
1. 在MATLAB中运行 `run_and_save_results.m`（我已经帮你写好了）
2. 在Python中使用 `compareTest_with_saved_matlab.py`
3. 完成！

### 🥈 备选：方案3（Octave）
**原因**:
- ✅ 完全免费
- ✅ 不需要Python 3.10
- ✅ 可以实时调用MATLAB代码

### 🥉 最后：方案1或2（Python 3.10虚拟环境）
**原因**:
- ⚠️ 配置复杂
- ⚠️ 需要额外安装Python 3.10
- ✅ 但是最灵活

## 快速开始（推荐方案4）

```bash
# 1. 在MATLAB中运行（只需一次）
# 打开MATLAB，运行:
run_and_save_results

# 2. 在Python中使用（你当前的Python环境）
python compareTest_with_saved_matlab.py
```

## 需要我帮你做什么？

我可以：
1. ✅ 创建完整的MATLAB批处理脚本
2. ✅ 创建Python读取脚本
3. ✅ 创建更新的对比测试脚本
4. ✅ 提供详细的使用说明

你想用哪个方案？我可以立即帮你实现！
