# grid_size优化快速参考卡

## 🎯 三种使用场景

### 场景1: 快速测试（推荐首次使用）
```bash
cd map_visualization
python optimize_grid_size_quick_test.py
```
- 测试单个区域（默认：长沙市-岳麓区）
- 快速验证功能
- 生成 `quick_test_result.png`

### 场景2: 优化特定区域
```python
# 编辑 optimize_grid_size_by_area.py
TARGET_CITY = "长沙市"
TARGET_AREA = "岳麓区"
```
```bash
python optimize_grid_size_by_area.py
```

### 场景3: 批量优化所有区域
```python
# 编辑 optimize_grid_size_by_area.py
TARGET_CITY = ""
TARGET_AREA = ""
```
```bash
python optimize_grid_size_by_area.py
```

## ⚙️ 关键参数速查

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `TARGET_CITY` | `""` | 目标城市 | 留空=所有城市 |
| `TARGET_AREA` | `""` | 目标区域 | 留空=所有区域 |
| `GRID_SIZE_START` | `0.01` | 起始值 | 密集区0.01，稀疏区0.05 |
| `GRID_SIZE_END` | `0.2` | 结束值 | 密集区0.1，稀疏区0.3 |
| `GRID_SIZE_STEP` | `0.01` | 步长 | 快速测试0.02，精确0.005 |
| `MAX_BUFFER_DISTANCE` | `5` | 缓冲距离 | 密集区3-5，稀疏区5-10 |
| `MIN_CLUSTER_SIZE` | `10` | 最小簇大小 | 细粒度5，粗粒度20 |
| `MIN_POINTS` | `50` | 最少点数 | 根据数据量调整 |

## 📊 轮廓系数速查

| 分数 | 评价 | 含义 |
|------|------|------|
| > 0.7 | ⭐⭐⭐ 优秀 | 聚类结构非常清晰 |
| 0.5-0.7 | ⭐⭐ 良好 | 聚类结构较清晰 |
| 0.25-0.5 | ⭐ 一般 | 聚类结构模糊 |
| < 0.25 | ❌ 较差 | 聚类结构不明显 |

## 📁 输出文件速查

```
grid_size_optimization_results/
├── optimization_results.json          # JSON格式结果
├── optimization_summary.md            # 汇总报告
├── optimization_summary.png           # 汇总图表（4子图）
└── {城市}_{区域}.png                  # 各区域曲线（2子图）
```

## 🔧 常见问题速查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无法计算轮廓系数 | 只有1个簇 | 增大`MAX_BUFFER_DISTANCE` |
| 优化时间太长 | 步长太小 | 增大`GRID_SIZE_STEP`到0.02 |
| 轮廓系数都很低 | 参数不合适 | 调整`MAX_BUFFER_DISTANCE` |
| 找不到区域 | 名称错误 | 查看`client_data_by_city/index.json` |

## 💡 推荐工作流程

```
1. 快速测试
   └─> python optimize_grid_size_quick_test.py

2. 单区域优化
   └─> 设置TARGET_CITY和TARGET_AREA
   └─> python optimize_grid_size_by_area.py

3. 查看结果
   └─> 打开 grid_size_optimization_results/
   └─> 查看图表和报告

4. 批量优化
   └─> 清空TARGET_CITY和TARGET_AREA
   └─> python optimize_grid_size_by_area.py

5. 应用推荐值
   └─> 使用optimization_summary.md中的推荐grid_size
```

## 📋 可用城市和区域

查看完整列表:
```bash
# 查看JSON文件
cat client_data_by_city/index.json

# 或在Python中
import json
with open('client_data_by_city/index.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for area in data['city_areas']:
        print(f"{area['city']} - {area['area']}: {area['count']}个客户")
```

主要城市:
- 长沙市 (3229个客户, 9个区)
- 株洲市 (1232个客户, 6个区)
- 湘潭市 (1093个客户, 5个区)
- 衡阳市 (487个客户, 6个区)
- 益阳市 (484个客户, 4个区)
- 常德市 (473个客户, 3个区)
- ...

## 🎨 图表说明

### 各区域图表 (城市_区域.png)
- **上图**: 轮廓系数 vs grid_size
  - 蓝色曲线: 轮廓系数变化
  - 红色虚线: 最佳grid_size位置
- **下图**: 簇数 vs grid_size
  - 绿色曲线: 簇数变化
  - 红色虚线: 最佳grid_size位置

### 汇总图表 (optimization_summary.png)
- **左上**: 最佳grid_size分布（直方图）
- **右上**: 轮廓系数分布（直方图）
- **左下**: grid_size vs 轮廓系数（散点图）
- **右下**: 客户数 vs grid_size（散点图，颜色=轮廓系数）

## 🚀 一键命令

```bash
# 进入目录
cd map_visualization

# 快速测试
python optimize_grid_size_quick_test.py

# 优化所有区域（客户数>=50）
python optimize_grid_size_by_area.py

# 查看结果
cd grid_size_optimization_results
ls -la
```

## 📞 相关文档

- **详细文档**: `grid_size优化使用说明.md`
- **完成状态**: `完成状态_grid_size优化.md`
- **数据分组**: `数据分组说明.md`

---

**提示**: 首次使用建议先运行快速测试脚本！
