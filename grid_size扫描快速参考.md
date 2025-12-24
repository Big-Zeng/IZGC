# Grid Size扫描 - 快速参考

## 🚀 一键运行

```bash
python test_grid_size_sweep.py
```

## ⚙️ 快速配置

```python
# 编辑 test_grid_size_sweep.py

# 基本配置
DATASET = "data/flame.csv"      # 数据集
GRID_SIZE_START = 0.1           # 起始值
GRID_SIZE_END = 3.0             # 结束值
GRID_SIZE_STEP = 0.1            # 步长

# 高级配置
MAX_BUFFER_DISTANCE = 5         # 缓冲距离
INTERPOLATION_METHOD = 'linear' # 插值方法
SHOW_VISUALIZATION = False      # 禁用可视化（加速）
```

## 📊 典型配置

### 快速测试（30秒）
```python
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.2    # 大步长
```

### 标准测试（1-2分钟）
```python
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.1    # 推荐 ✓
```

### 精细测试（2-4分钟）
```python
GRID_SIZE_START = 0.1
GRID_SIZE_END = 3.0
GRID_SIZE_STEP = 0.05   # 小步长
```

### 局部优化（40秒）
```python
GRID_SIZE_START = 0.5   # 已知最优值附近
GRID_SIZE_END = 0.8
GRID_SIZE_STEP = 0.02   # 很小步长
```

## 📈 输出结果

### 1. 控制台
```
[15/30] 测试 grid_size = 0.617... 完成 (ARI=0.8769, 用时=1.45s)

最佳 grid_size: 0.617
ARI:           0.8769  ← 最大值
```

### 2. 图表文件
`grid_size_sweep_results.png` - 4个子图展示ARI/AMI/FMI变化

### 3. 文本文件
`grid_size_sweep_results.txt` - 完整结果表格

## 🎯 使用流程

```
1. 配置参数 → 编辑脚本
2. 运行测试 → python test_grid_size_sweep.py
3. 查看结果 → 打开PNG和TXT文件
4. 使用最优值 → 在实际聚类中使用推荐的grid_size
```

## ⚡ 性能对比

| 步长 | 测试次数 | 时间 | 精度 |
|------|----------|------|------|
| 0.2  | 15次 | 30秒 | 粗略 |
| 0.1  | 30次 | 1分钟 | 良好 ✓ |
| 0.05 | 59次 | 2分钟 | 精细 |
| 0.02 | 146次 | 5分钟 | 最精细 |

## 💡 优化技巧

### 加速测试
```python
SHOW_VISUALIZATION = False  # 禁用可视化
GRID_SIZE_STEP = 0.2        # 增大步长
```

### 提高精度
```python
GRID_SIZE_STEP = 0.05       # 减小步长
```

### 两阶段策略
```python
# 阶段1：粗扫描（步长0.2）找到大致范围
# 阶段2：细扫描（步长0.02）精确定位
```

## 📝 结果示例

```
grid_size    ARI        AMI        FMI
----------------------------------------
0.100        0.6543     0.6123     0.6789
0.200        0.7234     0.6890     0.7456
0.300        0.7890     0.7456     0.8012
...
0.617        0.8769     0.8234     0.8912  ← 最佳
...
3.000        0.5234     0.4890     0.5456
```

## ❓ 常见问题

**Q: 如何选择范围？**  
A: 起始0.1，结束2-3倍平均点间距

**Q: 步长多少合适？**  
A: 推荐0.1，快速测试用0.2，精细用0.05

**Q: 测试太慢？**  
A: 设置 `SHOW_VISUALIZATION = False`

**Q: 最优值在边界？**  
A: 扩大测试范围重新运行

## 🔗 相关工具

- `optimize_grid_size.py` - 梯度下降优化（更快）
- `test_auto_grid_size.py` - 自动计算（最快）
- `compareTest2.py` - 单次测试

## 📌 推荐工作流

```
1. 快速测试 (步长0.2) → 找到大致范围
2. 标准测试 (步长0.1) → 确定最优值
3. 使用最优值 → 在实际应用中使用
```

---

**默认配置**: 0.1-3.0, 步长0.1  
**推荐时间**: 1-2分钟  
**输出文件**: PNG图表 + TXT结果
