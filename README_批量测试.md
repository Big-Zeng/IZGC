# 聚类算法批量测试系统

## 📋 概述

这是一个完整的聚类算法对比测试系统，支持自动遍历数据集、运行多个算法、生成对比报告。

## 🎯 主要功能

1. **批量测试** - 自动测试data文件夹中的所有数据集
2. **多算法对比** - KMeans、DBSCAN、Contour-Based、MATLAB Algorithm_PR
3. **自动优化** - 可选的grid_size自动优化
4. **详细报告** - 生成包含所有指标的文本报告
5. **统计分析** - 计算平均性能和标准差

## 📁 文件结构

```
项目根目录/
├── data/                           # 数据集文件夹
│   ├── flame.csv
│   ├── jain.csv
│   ├── Aggregation.csv
│   └── ...
├── results/                        # 结果输出文件夹（自动创建）
│   └── clustering_results.txt
├── batch_test_all_datasets.py      # 批量测试主程序
├── test_batch_quick.py             # 快速测试脚本
├── compareTest_with_matlab.py      # 单数据集测试（含MATLAB）
├── compareTest_template.py         # 单数据集测试（不含MATLAB）
├── optimize_grid_size.py           # grid_size优化器
├── clustering_metrics.py           # 评估指标
├── matlab_integration.py           # MATLAB集成
└── 批量测试使用说明.md             # 详细使用说明
```

## 🚀 快速开始

### 方法1: 快速测试（推荐新手）

测试前2个数据集，不含MATLAB，不优化：

```bash
python test_batch_quick.py
```

### 方法2: 完整测试

测试所有数据集，包含所有算法：

```bash
python batch_test_all_datasets.py
```

### 方法3: 单个数据集测试

测试单个数据集并可视化：

```bash
python compareTest_template.py
```

## ⚙️ 配置

编辑 `batch_test_all_datasets.py` 中的配置：

```python
ALGORITHMS_CONFIG = {
    'KMeans': {
        'enabled': True,        # ← 改为False禁用
        'params': {...}
    },
    'DBSCAN': {
        'enabled': True,
        'params': {...}
    },
    'Contour-Based': {
        'enabled': True,
        'params': {
            'optimize_grid': False  # ← 改为True启用优化
        }
    },
    'MATLAB-Algorithm_PR': {
        'enabled': True,        # ← 改为False跳过MATLAB
        'params': {...}
    }
}
```

## 📊 输出结果

### 结果文件

`results/clustering_results.txt` 包含：

1. **测试概要** - 测试时间、数据集数量、算法列表
2. **详细结果** - 每个数据集的完整指标
3. **汇总表格** - 所有结果的对比表
4. **平均性能** - 每个算法的统计指标

### 评估指标

- **AMI** (Adjusted Mutual Information) - 调整互信息
- **ARI** (Adjusted Rand Index) - 调整兰德指数
- **FMI** (F-Measure Index) - F度量指数
- **Silhouette** - 轮廓系数
- **Inertia** - 簇内误差平方和
- **Time** - 运行时间

## 🔧 算法说明

### 1. KMeans
- 经典K均值聚类
- 需要指定簇数
- 速度快，适合球形簇

### 2. DBSCAN
- 基于密度的聚类
- 自动发现簇数
- 可识别噪声点
- 参数：eps（邻域半径）、min_samples（最小样本数）

### 3. Contour-Based（你的算法）
- 基于等值线的聚类
- 自动计算grid_size
- 可选优化功能
- 参数：grid_size、max_buffer_distance

### 4. MATLAB Algorithm_PR
- MATLAB中的对比算法
- 需要MATLAB环境
- 参数：k（近邻数量）

## ⏱️ 运行时间

假设10个数据集，每个约200-300个点：

| 配置 | 预计时间 |
|------|----------|
| 快速测试（2个数据集，不含MATLAB） | 30秒 |
| 完整测试（不优化grid_size） | 3-5分钟 |
| 完整测试（优化grid_size） | 30-50分钟 |

## 💡 使用建议

### 首次使用
1. 运行 `python test_batch_quick.py` 快速测试
2. 检查 `results/test_results.txt` 确认正常
3. 运行完整测试 `python batch_test_all_datasets.py`

### 日常使用
- **快速对比**: 禁用MATLAB和优化，只测试主要算法
- **完整评估**: 启用所有算法，但不优化
- **最优参数**: 启用优化，但只测试关键数据集

### 性能优化
- 禁用不需要的算法
- 设置 `optimize_grid=False`
- 减少测试数据集数量
- 调整DBSCAN的eps和min_samples

## 📝 示例输出

```
================================================================================
聚类算法批量测试结果
================================================================================
测试时间: 2024-12-21 15:30:45
数据集数量: 10
算法: KMeans, DBSCAN, Contour-Based, MATLAB-Algorithm_PR
================================================================================

================================================================================
数据集: flame.csv
================================================================================
样本数: 240
真实簇数: 2
--------------------------------------------------------------------------------

算法: Contour-Based
--------------------------------------------------
  发现簇数: 2
  噪声点数: 3
  grid_size: 0.617
  AMI:        0.8234
  ARI:        0.8769
  FMI:        0.8912
  Silhouette: 0.7234
  Inertia:    567.89
  运行时间:   2.34s

...

================================================================================
汇总对比表
================================================================================

数据集               算法                 ARI        AMI        FMI        Silhouette   时间(s)
----------------------------------------------------------------------------------------------------
flame.csv            KMeans               0.6789     0.5234     0.7123     0.4567       0.05
flame.csv            Contour-Based        0.8769     0.8234     0.8912     0.7234       2.34
...

平均性能指标
----------------------------------------------------------------------------------------------------

Contour-Based:
  平均 ARI: 0.8234 (±0.0987)
  平均 AMI: 0.7890 (±0.1123)
  平均 FMI: 0.8456 (±0.0876)
  平均时间: 2.45s (±0.67s)
```

## ❓ 常见问题

### Q: MATLAB算法失败？
**A**: 设置 `'enabled': False` 跳过MATLAB算法

### Q: 如何加快速度？
**A**: 禁用不需要的算法，设置 `optimize_grid=False`

### Q: 如何只测试某些数据集？
**A**: 将其他CSV文件移出data文件夹

### Q: 结果文件在哪？
**A**: `results/clustering_results.txt`

### Q: 如何修改算法参数？
**A**: 编辑 `ALGORITHMS_CONFIG` 中的 `params`

## 🔗 相关文档

- `批量测试使用说明.md` - 详细使用说明
- `OPTIMIZE_GRID_SIZE_GUIDE.md` - grid_size优化指南
- `使用说明_优化grid_size.md` - 优化功能中文说明
- `AUTO_GRID_SIZE_GUIDE.md` - 自动grid_size计算说明

## 📞 技术支持

如遇问题，请检查：
1. 数据集格式是否正确（必须有x, y, target列）
2. 是否安装了所需的Python包
3. MATLAB环境是否配置正确（如果使用MATLAB算法）
4. 查看错误信息和日志

---

**快速开始**: `python test_batch_quick.py`  
**完整测试**: `python batch_test_all_datasets.py`  
**查看结果**: 打开 `results/clustering_results.txt`
