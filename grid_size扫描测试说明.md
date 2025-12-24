# Grid Size扫描测试说明

## 功能说明

`test_grid_size_sweep.py` 通过遍历不同的grid_size值，找到使ARI（调整兰德指数）最大的最优grid_size配置。

## 快速开始

### 基本使用

```bash
python test_grid_size_sweep.py
```

### 配置参数

编辑脚本中的配置区域：

```python
# 配置区域
DATASET = "data/flame.csv"              # 数据集路径
GRID_SIZE_START = 0.1                   # 起始grid_size
GRID_SIZE_END = 3.0                     # 结束grid_size
GRID_SIZE_STEP = 0.1                    # 步长
MAX_BUFFER_DISTANCE = 5                 # 最大缓冲距离
INTERPOLATION_METHOD = 'linear'         # 插值方法

# 是否显示可视化（设为False加快速度）
SHOW_VISUALIZATION = False
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `GRID_SIZE_START` | `0.1` | 起始grid_size | 0.01-0.5 |
| `GRID_SIZE_END` | `3.0` | 结束grid_size | 2.0-5.0 |
| `GRID_SIZE_STEP` | `0.1` | 步长 | 0.05-0.2 |
| `MAX_BUFFER_DISTANCE` | `5` | 最大缓冲距离 | 3-10 |
| `INTERPOLATION_METHOD` | `'linear'` | 插值方法 | linear/cubic/idw |

### 调整建议

**快速测试**（粗略扫描）：
```python
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.2    # 较大步长
```

**精细测试**（精确扫描）：
```python
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.05   # 较小步长
```

**局部优化**（在已知范围内细化）：
```python
GRID_SIZE_START = 0.5   # 已知最优值附近
GRID_SIZE_END = 0.8
GRID_SIZE_STEP = 0.02   # 很小的步长
```

## 输出结果

### 1. 控制台输出

```
================================================================================
Grid Size扫描测试
================================================================================

数据集: data/flame.csv
样本数: 240
真实簇数: 2
X范围: [0.00, 15.00]
Y范围: [0.00, 10.00]

开始扫描 grid_size 从 0.1 到 3.0，步长 0.1
插值方法: linear
最大缓冲距离: 5
================================================================================

[1/30] 测试 grid_size = 0.100... 完成 (ARI=0.6543, 用时=2.34s)
[2/30] 测试 grid_size = 0.200... 完成 (ARI=0.7234, 用时=2.12s)
[3/30] 测试 grid_size = 0.300... 完成 (ARI=0.7890, 用时=1.98s)
...
[15/30] 测试 grid_size = 1.500... 完成 (ARI=0.8769, 用时=1.45s)
...

================================================================================
最佳结果
================================================================================
最佳 grid_size: 0.617
ARI:           0.8769
AMI:           0.8234
FMI:           0.8912
Silhouette:    0.7234
发现簇数:      2
噪声点数:      3
================================================================================
```

### 2. 结果图表

自动生成 `grid_size_sweep_results.png`，包含4个子图：
- **ARI vs grid_size** - ARI随grid_size的变化
- **AMI vs grid_size** - AMI随grid_size的变化
- **FMI vs grid_size** - FMI随grid_size的变化
- **所有指标对比** - 三个指标的综合对比

### 3. 结果文件

自动生成 `grid_size_sweep_results.txt`，包含：
- 测试配置信息
- 所有grid_size的详细结果表格
- 最佳结果汇总

示例内容：
```
================================================================================
Grid Size扫描测试结果
================================================================================
数据集: data/flame.csv
grid_size范围: 0.1 - 3.0, 步长: 0.1
插值方法: linear
最大缓冲距离: 5
测试数量: 30
================================================================================

详细结果
--------------------------------------------------------------------------------
grid_size    簇数     噪声     ARI        AMI        FMI        Silhouette   时间(s)
--------------------------------------------------------------------------------
0.100        3        5        0.6543     0.6123     0.6789     0.5234       2.34
0.200        2        4        0.7234     0.6890     0.7456     0.6123       2.12
0.300        2        3        0.7890     0.7456     0.8012     0.6789       1.98
...
0.617        2        3        0.8769     0.8234     0.8912     0.7234       1.45  ← 最佳
...
================================================================================

最佳结果（ARI最大）
--------------------------------------------------------------------------------
最佳 grid_size: 0.617
ARI:           0.8769
AMI:           0.8234
FMI:           0.8912
Silhouette:    0.7234
发现簇数:      2
噪声点数:      3
================================================================================
```

## 使用场景

### 场景1: 寻找最优grid_size

```python
# 配置
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.1

# 运行
python test_grid_size_sweep.py

# 结果：找到最优grid_size = 0.617
```

### 场景2: 对比不同插值方法

```python
# 测试linear
INTERPOLATION_METHOD = 'linear'
python test_grid_size_sweep.py

# 测试cubic
INTERPOLATION_METHOD = 'cubic'
python test_grid_size_sweep.py

# 测试idw
INTERPOLATION_METHOD = 'idw'
python test_grid_size_sweep.py

# 对比三个结果文件，选择最佳方法
```

### 场景3: 精细调优

```python
# 第一步：粗略扫描
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.2
# 结果：最优值在0.6附近

# 第二步：精细扫描
GRID_SIZE_START = 0.4
GRID_SIZE_END = 0.8
GRID_SIZE_STEP = 0.02
# 结果：精确最优值 = 0.617
```

## 运行时间估算

假设数据集有200-300个点：

| 配置 | 测试次数 | 预计时间 |
|------|----------|----------|
| 步长0.2 (0.1-3.0) | 15次 | 30-45秒 |
| 步长0.1 (0.1-3.0) | 30次 | 1-2分钟 |
| 步长0.05 (0.1-3.0) | 59次 | 2-4分钟 |
| 步长0.02 (0.4-0.8) | 21次 | 40-60秒 |

## 性能优化建议

### 1. 加快测试速度

```python
# 禁用可视化
SHOW_VISUALIZATION = False

# 增大步长
GRID_SIZE_STEP = 0.2

# 缩小范围
GRID_SIZE_START = 0.3
GRID_SIZE_END = 1.5
```

### 2. 提高精度

```python
# 减小步长
GRID_SIZE_STEP = 0.05

# 扩大范围
GRID_SIZE_START = 0.01
GRID_SIZE_END = 5.0
```

### 3. 两阶段策略

```python
# 阶段1：粗略扫描（快速）
GRID_SIZE_STEP = 0.2
# 找到大致范围

# 阶段2：精细扫描（精确）
GRID_SIZE_START = 最优值 - 0.2
GRID_SIZE_END = 最优值 + 0.2
GRID_SIZE_STEP = 0.02
```

## 常见问题

### Q1: 如何选择起始和结束值？

**A**: 
- 起始值：通常从0.1开始（太小会导致计算量大）
- 结束值：根据数据范围，通常2-5倍的平均点间距
- 可以先用自动计算的grid_size作为参考

### Q2: 步长如何选择？

**A**: 
- 粗略扫描：0.1-0.2
- 正常扫描：0.05-0.1
- 精细扫描：0.01-0.05

### Q3: 测试太慢怎么办？

**A**: 
1. 设置 `SHOW_VISUALIZATION = False`
2. 增大步长
3. 缩小测试范围
4. 使用两阶段策略

### Q4: 如何解读结果？

**A**: 
- 查看ARI曲线，找到峰值对应的grid_size
- 如果有多个峰值，选择ARI最高的
- 参考AMI和FMI确认结果的稳定性

### Q5: 最优grid_size在边界怎么办？

**A**: 
- 如果在起始边界：减小GRID_SIZE_START
- 如果在结束边界：增大GRID_SIZE_END
- 重新运行测试

## 高级用法

### 自定义评估指标

修改 `find_best_grid_size()` 方法：

```python
# 使用AMI而不是ARI
sorted_results = sorted(self.results, key=lambda x: x['AMI'], reverse=True)

# 或使用综合指标
sorted_results = sorted(self.results, 
                       key=lambda x: (x['ARI'] + x['AMI'] + x['FMI']) / 3, 
                       reverse=True)
```

### 批量测试多个数据集

```python
datasets = ['data/flame.csv', 'data/jain.csv', 'data/Aggregation.csv']

for dataset in datasets:
    DATASET = dataset
    # 运行测试
    # ...
```

## 相关文件

- `test_grid_size_sweep.py` - 主测试脚本
- `optimize_grid_size.py` - 梯度下降优化（另一种方法）
- `test_auto_grid_size.py` - 自动grid_size计算测试
- `compareTest2.py` - 原始对比测试脚本

## 对比其他方法

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Grid Size扫描** | 全局搜索，找到真正最优值 | 较慢，需要多次测试 | 追求最优结果 |
| **梯度下降优化** | 快速，自动调整 | 可能陷入局部最优 | 快速优化 |
| **自动计算** | 最快，无需测试 | 可能不是最优 | 快速开始 |

---

**快速开始**: `python test_grid_size_sweep.py`  
**查看结果**: 打开 `grid_size_sweep_results.txt` 和 `grid_size_sweep_results.png`  
**推荐配置**: 步长0.1，范围0.1-3.0
