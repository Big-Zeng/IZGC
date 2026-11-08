# Contour-Based Clustering

`compareTest2.py` 是等值线聚类算法的单数据集启动入口。它读取带真实标签的二维点集，依次运行 **Contour-Based Clustering**、**KMeans** 和 **DBSCAN**，输出聚类指标与对比可视化结果。

## 算法流程

1. 读取 CSV 数据，提取坐标 `x, y` 和真实标签 `target`。
2. 可选地将坐标按轴归一化到 `[0, 100]`。
3. 使用 `cKDTree` 统计每个点在给定半径内的邻居数，作为该点的密度值。
4. 在坐标范围外扩展网格，生成虚拟点；对密度场做线性插值。
5. 提取 10 条等值线，将等值线转换为多边形区域，并建立父子层次结构。
6. 将最外层区域作为密度中心，把实际点分配到各中心；不足 `min_cluster_size` 的小簇按配置合并。
7. 计算 Inertia、Silhouette、ARI（以及 AMI/FMI），生成四联对比图并保存。

## 文件说明

| 文件 | 作用 |
|------|------|
| `compareTest2.py` | 启动入口，完成数据读取、三种算法对比、指标计算和绘图 |
| `ContourBasedClusteringWithoutZ22.py` | 等值线聚类算法实现 |
| `clustering_metrics.py` | AMI、ARI、FMI 等聚类指标实现 |
| `data/` | 测试数据集 |

运行后会在仓库根目录生成：

- `contour_areas.png`：密度场等值线区域图（600 DPI）
- `cluster_result.png`：等值线聚类结果图（600 DPI）
- `clustering_comparison.png`：Ground Truth / KMeans / DBSCAN / Contour 四联对比图（150 DPI）

## 环境依赖

- Python 3.9+
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- shapely

安装：

```bash
pip install numpy pandas scikit-learn scipy matplotlib shapely
```

交互模式需要 matplotlib 的图形后端（默认尝试 `TkAgg`）。Linux 下如缺少 Tk：

```bash
sudo apt install python3-tk
```

## 快速开始

在仓库根目录执行：

```bash
python compareTest2.py
```

入口文件使用相对路径读取数据和保存图片，因此建议始终从仓库根目录运行。

## 输入数据格式

CSV 至少包含以下三列：

```csv
x,y,target
1.85,27.8,1
1.35,26.65,1
1.40,23.25,2
```

- `x`、`y`：二维坐标，数值类型。
- `target`：真实簇标签，整数或字符串均可，但需要能作为分类标签使用。

`data/real_data/` 中的部分真实数据只有 `x,y` 两列。直接用于本脚本前需要补充 `target` 列，否则会在 `df['target']` 处报错。

## 配置说明

### 数据集与归一化

在 `compareTest2.py` 顶部修改：

```python
DATASET = "data/flame.csv"   # 数据集路径
NORMALIZE_COORDS = True      # 是否将坐标归一化到 [0, 100]
```

归一化按 X、Y 两个轴分别进行。真实地理坐标（经纬度跨度较小）建议开启归一化，否则密度半径和 DBSCAN 的 `eps` 需要重新调整。

### 对比算法参数

KMeans：

```python
n_clusters = len(np.unique(y_true))  # 使用真实簇数
random_state = 42
n_init = 10
```

DBSCAN：

```python
eps = 5.0
min_samples = 5
metric = 'euclidean'
```

### 等值线聚类参数

入口通过 `VRPAnalyzer.run_contour_clustering()` 调用算法，当前默认值为：

```python
grid_size = 3.160
max_buffer_distance = 0
min_cluster_size = 2   # 注意：构造函数中实际传入的是 5
radius = 30
```

其中的实际行为是：

- `radius`：密度估计的邻域半径，影响每个点的密度值。
- `grid_size`：虚拟点网格大小。值越小，网格越密、虚拟点和插值计算量越大，聚类区域边界越精细。
- `max_buffer_distance`：区域缓冲区扩展距离，用于把边界附近的点分配给密度中心。
- `min_cluster_size`：小于该点数的簇会被合并到最近的大簇。当前构造函数中写死为 `5`，函数参数 `min_cluster_size` 实际未生效。
- `merge_small_clusters=True`：小簇合并到大簇；改为 `False` 时小簇点会被标记为噪声（标签 `-1`）。

如需修改有效的最小簇大小，请同时修改 `compareTest2.py` 中构造实例处的这一行：

```python
min_cluster_size=5,
```

## 交互模式

`ContourBasedClusteringWithoutZ22` 的 `interactive` 默认是 `True`，入口没有显式覆盖，因此运行过程中会依次弹出窗口：

1. 网格与虚拟点窗口：可在空白处用鼠标左键添加零值点，关闭窗口后继续。
2. 等值线区域窗口：关闭后继续。
3. 聚类结果窗口：关闭后继续。
4. 最后 `compareTest2.py` 自身还会调用一次 `plt.show()` 显示四联对比图。

批量运行或服务器无图形界面时，需要关闭交互。可在创建算法实例时增加 `interactive=False`：

```python
cluster = ContourBasedClusteringWithoutZ22.ContourBasedClusteringWithoutZ22(
    combined_data,
    grid_size=grid_size,
    max_buffer_distance=max_buffer_distance,
    min_cluster_size=5,
    merge_small_clusters=True,
    interactive=False
)
```

并把文件末尾的 `plt.show()` 删除或改为保存后关闭：

```python
# plt.show()
```

也可以使用无界面后端：

```bash
MPLBACKEND=Agg python compareTest2.py
```

需要注意：`MPLBACKEND=Agg` 下 `plt.show()` 不会阻塞，但也不会显示窗口；仍建议设置 `interactive=False`，以便及时关闭图形并避免无界面环境下产生多余窗口对象。

## 输出指标

控制台会打印每个算法的：

- **Inertia**：簇内误差平方和，越小通常表示簇内越紧凑。
- **Silhouette**：轮廓系数，范围 `[-1, 1]`，越大越好。
- **ARI**：调整兰德指数，范围 `[-1, 1]`，1 表示与真实标签完全一致。

入口还调用了 `ClusteringMetrics.evaluate_all()`，可计算：

- **AMI**：调整互信息
- **ARI**：调整兰德指数
- **FMI**：Fowlkes-Mallows 指数

目前 `evaluate_all()` 的结果保存在局部变量 `metrics` 中，尚未打印。如需输出 AMI/FMI，可在 `run_contour_clustering()` 中补充：

```python
metrics = ClusteringMetrics.evaluate_all(y_true, pre_labels)
print(f"AMI: {metrics['AMI']:.4f}, ARI: {metrics['ARI']:.4f}, FMI: {metrics['FMI']:.4f}")
```

## 参数调优建议

1. **先确定坐标尺度**：优先开启 `NORMALIZE_COORDS=True`，让不同数据集的 `radius`、`eps` 有一致的参考范围。
2. **调 `radius`**：半径过小会导致密度过于稀疏、区域碎片化；过大会把相邻簇连成一片。
3. **调 `grid_size`**：从 `3.0` 左右开始尝试；网格过细会显著增加运行时间，过粗会丢失簇边界。
4. **调 `min_cluster_size`**：数据噪声较多或存在大量小簇时适当增大，可减少碎片簇。
5. **调 `eps`**：DBSCAN 的 `eps` 与坐标尺度直接相关，归一化后可从 `3~8` 范围尝试。

## 常见问题

**`KeyError: 'target'`**

数据文件缺少真实标签列。请补充 `target` 列，或修改脚本以适配无标签数据（此时 ARI 等有监督指标无法计算）。

**窗口卡住不继续**

算法默认开启交互模式，需要手动关闭每个 matplotlib 窗口。批量运行时按上文设置 `interactive=False`。

**中文显示为方框**

算法文件设置了 `SimHei` 字体。系统未安装该字体时会出现警告，可安装中文字体或修改 `plt.rcParams['font.sans-serif']`。

**运行很慢或内存占用高**

通常是 `grid_size` 过小或数据量过大导致虚拟点和插值网格数量激增。适当增大 `grid_size`，或对大数据集先做降采样。

**没有生成图片**

确认脚本从仓库根目录运行，并检查当前目录是否可写；图片默认保存在仓库根目录。
