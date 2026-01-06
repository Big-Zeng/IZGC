# grid_size优化使用说明

## 📊 功能说明

`optimize_grid_size_by_area.py` 脚本用于对每个区域的客户数据进行grid_size参数优化，找到使**轮廓系数（Silhouette Score）**最佳的grid_size值。

## 🎯 评价指标

**轮廓系数 (Silhouette Score)**:
- 范围: [-1, 1]
- 值越大越好
- > 0.7: 优秀
- 0.5 - 0.7: 良好
- 0.25 - 0.5: 一般
- < 0.25: 较差

轮廓系数衡量：
- 簇内紧密度（点与同簇其他点的距离）
- 簇间分离度（点与其他簇点的距离）

## ⚙️ 参数配置

### 区域选择参数

```python
TARGET_CITY = ""            # 指定城市名称，如 "长沙市"
TARGET_AREA = ""            # 指定区域名称，如 "岳麓区"
```

**使用方式**:
1. **处理所有区域**: 两个参数都留空
2. **处理特定城市**: 设置 `TARGET_CITY = "长沙市"`
3. **处理特定区域**: 设置 `TARGET_CITY = "长沙市"` 和 `TARGET_AREA = "岳麓区"`

### grid_size扫描参数

```python
GRID_SIZE_START = 0.01      # grid_size起始值
GRID_SIZE_END = 0.2         # grid_size结束值
GRID_SIZE_STEP = 0.01       # grid_size步长
```

**说明**:
- 起始值: 建议从0.01开始
- 结束值: 建议不超过0.2
- 步长: 0.01较快，0.005更精确但更慢

### 聚类参数

```python
MAX_BUFFER_DISTANCE = 5     # 最大缓冲距离
MIN_CLUSTER_SIZE = 10       # 最小簇大小
MIN_POINTS = 50             # 最少点数（少于此值的区域跳过）
```

## 🚀 使用方法

### 1. 优化所有区域

```bash
cd map_visualization
python optimize_grid_size_by_area.py
```

### 2. 优化特定城市

编辑 `optimize_grid_size_by_area.py`:
```python
TARGET_CITY = "长沙市"
TARGET_AREA = ""
```

运行:
```bash
python optimize_grid_size_by_area.py
```

### 3. 优化特定区域

编辑 `optimize_grid_size_by_area.py`:
```python
TARGET_CITY = "长沙市"
TARGET_AREA = "岳麓区"
```

运行:
```bash
python optimize_grid_size_by_area.py
```

## 📁 输出文件

### 目录结构

```
grid_size_optimization_results/
├── optimization_results.json          # 所有结果（JSON格式）
├── optimization_summary.md            # 汇总报告（Markdown）
├── optimization_summary.png           # 汇总图表
├── 长沙市_岳麓区.png                  # 各区域优化曲线
├── 长沙市_雨花区.png
└── ... (其他区域)
```

### JSON结果格式

```json
[
  {
    "city": "长沙市",
    "area": "岳麓区",
    "count": 762,
    "best_grid_size": 0.05,
    "best_silhouette_score": 0.6234,
    "best_n_clusters": 8,
    "best_n_noise": 0,
    "all_results": [
      {
        "grid_size": 0.01,
        "n_clusters": 15,
        "n_noise": 5,
        "silhouette_score": 0.4521
      },
      ...
    ]
  },
  ...
]
```

### 汇总报告内容

1. **参数设置**: 显示所有配置参数
2. **优化结果表**: 每个区域的最佳grid_size和轮廓系数
3. **统计分析**: 
   - 最佳grid_size的平均值、中位数、标准差
   - 轮廓系数的平均值、中位数、范围
4. **推荐参数**: 基于所有区域的中位数

### 图表说明

**各区域图表** (如 `长沙市_岳麓区.png`):
- 上图: 轮廓系数 vs grid_size
- 下图: 簇数 vs grid_size
- 红色虚线: 最佳grid_size位置

**汇总图表** (`optimization_summary.png`):
- 左上: 最佳grid_size分布直方图
- 右上: 轮廓系数分布直方图
- 左下: grid_size vs 轮廓系数散点图
- 右下: 客户数 vs 最佳grid_size（颜色表示轮廓系数）

## 📊 使用示例

### 示例1: 优化长沙市岳麓区

```python
# 编辑 optimize_grid_size_by_area.py
TARGET_CITY = "长沙市"
TARGET_AREA = "岳麓区"
GRID_SIZE_START = 0.01
GRID_SIZE_END = 0.2
GRID_SIZE_STEP = 0.01
```

运行后输出:
```
================================================================================
[1/1] 处理: 长沙市 - 岳麓区 (762个客户)
================================================================================

开始扫描grid_size (20个值)...
  grid_size=0.010: 簇数=15, 噪声=5, 轮廓系数=0.4521
  grid_size=0.020: 簇数=12, 噪声=3, 轮廓系数=0.5234
  ...
  grid_size=0.050: 簇数=8, 噪声=0, 轮廓系数=0.6234
  ...

✓ 最佳结果:
  grid_size: 0.050
  轮廓系数: 0.6234
  簇数: 8
  噪声点: 0
```

### 示例2: 批量优化所有区域

```python
# 编辑 optimize_grid_size_by_area.py
TARGET_CITY = ""
TARGET_AREA = ""
MIN_POINTS = 100  # 只处理客户数>=100的区域
```

运行后会处理所有符合条件的区域，生成汇总报告。

## 🎯 参数调优建议

### grid_size范围选择

1. **数据密集区域**（如市中心）:
   - 起始值: 0.01
   - 结束值: 0.1
   - 步长: 0.005

2. **数据稀疏区域**（如郊区）:
   - 起始值: 0.05
   - 结束值: 0.3
   - 步长: 0.01

3. **快速测试**:
   - 起始值: 0.01
   - 结束值: 0.2
   - 步长: 0.02

### MAX_BUFFER_DISTANCE

- **密集区域**: 3-5
- **稀疏区域**: 5-10
- **效果**: 值越大，簇的覆盖范围越大

### MIN_CLUSTER_SIZE

- **需要细粒度聚类**: 5
- **一般情况**: 10
- **需要粗粒度聚类**: 20

## 📝 结果解读

### 轮廓系数解读

| 轮廓系数 | 评价 | 说明 |
|---------|------|------|
| > 0.7 | 优秀 | 聚类结构非常清晰 |
| 0.5 - 0.7 | 良好 | 聚类结构较清晰 |
| 0.25 - 0.5 | 一般 | 聚类结构模糊 |
| < 0.25 | 较差 | 聚类结构不明显 |

### 簇数分析

- **簇数过多**: 可能grid_size太小，增大grid_size
- **簇数过少**: 可能grid_size太大，减小grid_size
- **理想簇数**: 根据业务需求，通常3-15个簇较合理

### 噪声点分析

- **噪声点多**: 增大MAX_BUFFER_DISTANCE
- **无噪声点**: 理想情况，所有点都被分配
- **少量噪声**: 正常情况，可能是离群点

## 🔍 常见问题

### Q1: 所有grid_size都无法计算轮廓系数

**原因**: 
- 只产生1个簇（无法计算轮廓系数）
- 所有点都是噪声

**解决**:
- 增大MAX_BUFFER_DISTANCE
- 调整grid_size范围
- 减小MIN_CLUSTER_SIZE

### Q2: 优化时间太长

**原因**: grid_size步长太小或范围太大

**解决**:
- 增大GRID_SIZE_STEP（如0.01 → 0.02）
- 缩小范围（如0.01-0.2 → 0.03-0.15）
- 只处理特定区域

### Q3: 不同区域的最佳grid_size差异很大

**原因**: 不同区域的客户密度和分布不同

**解决**:
- 这是正常现象
- 使用汇总报告中的推荐值（中位数）作为通用参数
- 或为不同区域使用不同的grid_size

### Q4: 轮廓系数都很低（< 0.3）

**原因**: 
- 数据本身聚类结构不明显
- 参数设置不合适

**解决**:
- 调整MAX_BUFFER_DISTANCE
- 尝试不同的MIN_CLUSTER_SIZE
- 检查数据质量

## 💡 最佳实践

1. **首次使用**: 
   - 选择一个代表性区域（如主城区）
   - 使用默认参数快速测试
   - 根据结果调整参数

2. **批量优化**:
   - 先处理客户数较多的区域（MIN_POINTS=100）
   - 分析结果，找出规律
   - 再处理所有区域

3. **参数选择**:
   - 优先使用汇总报告中的推荐值（中位数）
   - 对特殊区域使用其专属的最佳值
   - 定期重新优化（数据变化时）

4. **结果验证**:
   - 查看生成的图表，确认曲线合理
   - 检查轮廓系数是否在合理范围
   - 对比不同区域的结果

## 📞 相关文件

- `optimize_grid_size_by_area.py` - 优化脚本
- `split_data_by_city.py` - 数据分组脚本
- `client_data_by_city/index.json` - 区域索引
- `test_grid_size_sweep.py` - 参考实现

---

**更新时间**: 2025-12-31  
**评价指标**: 轮廓系数 (Silhouette Score)  
**适用场景**: 区域级客户聚类分析
