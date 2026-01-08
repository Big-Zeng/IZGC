# 聚类算法对比模块

## 📁 文件结构

```
algorithm_comparison/
├── __init__.py           # 模块初始化
├── utils.py              # 通用工具函数
├── alg_contour.py        # ContourBasedClustering
├── alg_kmeans.py         # K-means++
├── alg_dbscan.py         # DBSCAN
├── alg_clique.py         # CLIQUE
├── alg_sting.py          # STING
├── alg_dpc_fwsn.py       # DPC-FWSN (MATLAB)
├── run_all.py            # 运行所有算法对比
├── results/              # 输出结果目录
└── README.md             # 本文件
```

## 🚀 快速开始

### 运行所有算法对比

```bash
cd algorithm_comparison
python run_all.py
```

### 运行单个算法

```bash
# ContourBasedClustering
python alg_contour.py

# K-means++
python alg_kmeans.py

# DBSCAN
python alg_dbscan.py

# CLIQUE
python alg_clique.py

# STING
python alg_sting.py

# DPC-FWSN (需要MATLAB)
python alg_dpc_fwsn.py
```

## ⚙️ 配置参数

每个算法文件顶部都有配置区域：

```python
# ==================== 配置 ====================
DATASET = "../data/Compound.csv"    # 数据集路径
OUTPUT_DIR = "results"               # 输出目录
NORMALIZE = True                     # 是否归一化

# 参数搜索范围
PARAM_GRID = {
    'param1': [value1, value2, ...],
    'param2': [value1, value2, ...],
}

# 优化目标: 'ARI', 'AMI', 'FMI', 'Silhouette'
OPTIMIZE_METRIC = 'ARI'
# ==============================================
```

## 📊 输出文件

### 图表 (PNG, 300 DPI)

- `01_GroundTruth_{数据集}.png` - 真实标签
- `02_ContourBased_{数据集}.png` - 等值线聚类
- `03_KMeans_{数据集}.png` - K-means++
- `04_DBSCAN_{数据集}.png` - DBSCAN
- `05_CLIQUE_{数据集}.png` - CLIQUE
- `06_STING_{数据集}.png` - STING
- `07_DPC-FWSN_{数据集}.png` - DPC-FWSN

### 结果文件 (CSV)

- `comparison_results_{数据集}.csv` - 所有算法的对比结果

## 📈 评价指标

| 指标 | 说明 | 范围 |
|------|------|------|
| AMI | 调整互信息 | [-1, 1] |
| ARI | 调整兰德指数 | [-1, 1] |
| FMI | FM指数 | [0, 1] |
| Silhouette | 轮廓系数 | [-1, 1] |

## 🎨 图表设置

论文级别的图表设置：
- 分辨率: 300 DPI
- 字体: Times New Roman
- 图表大小: 8×6 英寸
- 点大小: 30
- 颜色: tab10 色板

## 💡 使用建议

### 1. 修改数据集

编辑各算法文件中的 `DATASET` 变量：

```python
DATASET = "../data/jain.csv"
```

### 2. 调整参数范围

根据数据集特点调整 `PARAM_GRID`：

```python
# 小数据集
PARAM_GRID = {
    'grid_size': [1.0, 2.0, 3.0],
    ...
}

# 大数据集
PARAM_GRID = {
    'grid_size': [5.0, 10.0, 15.0],
    ...
}
```

### 3. 更改优化目标

```python
OPTIMIZE_METRIC = 'Silhouette'  # 使用轮廓系数优化
```

## 🔧 依赖

- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

MATLAB算法额外需要：
- MATLAB (R2022或更高)
- MATLAB Engine for Python

## 📝 算法说明

### ContourBasedClustering
基于密度等值线的聚类算法，通过密度场的等值线划分区域。

### K-means++
经典K均值聚类的改进版，使用更好的初始化策略。

### DBSCAN
基于密度的空间聚类算法，能发现任意形状的簇。

### CLIQUE
基于网格的子空间聚类算法，将空间划分为网格单元。

### STING
基于网格的统计信息聚类算法，使用网格单元的统计信息。

### DPC-FWSN
密度峰值聚类算法的变体，通过MATLAB实现。

---

**创建日期**: 2025-01-04
