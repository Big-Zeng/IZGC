# 交互式聚类分析系统 - Leaflet版本使用说明

## 🎯 新版本特性

### 与百度地图版本的区别

| 特性 | Leaflet版本 | 百度地图版本 |
|------|------------|-------------|
| 地图服务 | OpenStreetMap（免费） | 百度地图（需要AK） |
| 等值线显示 | ✓ 显示聚类边界多边形 | ✗ 不显示 |
| 点颜色 | ✓ 按簇区分颜色 | ✓ 按簇区分颜色 |
| 网络要求 | 需要访问国际网络 | 国内网络即可 |
| 配置要求 | 无需配置 | 需要百度地图AK |

## 🚀 快速启动

### 1. 启动服务器

```bash
cd map_visualization
python clustering_server.py
```

### 2. 访问系统

打开浏览器访问：**http://localhost:5000**

系统会自动加载 Leaflet 版本的交互式聚类页面。

## 🎨 界面说明

### 左侧面板
- **参数调整区**: 三个滑块调整聚类参数
  - grid_size: 网格大小（0.01-0.2）
  - max_buffer_distance: 最大缓冲距离（1-20）
  - min_cluster_size: 最小簇大小（1-50）
- **聚类按钮**: 点击执行聚类分析
- **结果显示区**: 显示聚类结果和评价指标

### 右侧地图
- **彩色多边形**: 等值线聚类边界（不同颜色代表不同簇）
- **彩色圆点**: 客户位置（颜色与所属簇一致）
- **信息框**: 左上角显示聚类统计
- **图例**: 右下角显示各簇颜色和点数

## 🔍 功能详解

### 1. 等值线边界显示

**特点**:
- 每个簇显示为一个彩色多边形区域
- 多边形边界表示聚类的等值线轮廓
- 填充半透明，可以看到下层的地图

**交互**:
- 点击多边形：显示簇信息（簇编号、点数、中心坐标）
- 鼠标悬停：高亮显示

### 2. 客户点显示

**特点**:
- 圆形标记，颜色与所属簇一致
- 大小适中，不会遮挡地图

**交互**:
- 点击标记：显示客户详细信息
- 鼠标悬停：显示客户名称和簇编号

### 3. 颜色编码

**规则**:
- 每个簇自动分配一个独特的颜色
- 使用HSL色彩空间均匀分布
- 噪声点（如果有）显示为灰色

## 📊 使用流程

```
1. 调整参数
   ↓
2. 点击"开始聚类分析"
   ↓
3. 等待10-30秒计算
   ↓
4. 地图显示：
   - 彩色多边形（聚类边界）
   - 彩色圆点（客户位置）
   ↓
5. 查看评价指标
   ↓
6. 交互探索：
   - 点击多边形查看簇信息
   - 点击圆点查看客户信息
   ↓
7. 调整参数重新聚类
```

## 🎯 参数调优

### 场景1：边界太复杂
- ✓ 增大 `grid_size`（如 0.05 → 0.1）
- ✓ 减少等值线层级（算法内部参数）

### 场景2：边界太简单
- ✓ 减小 `grid_size`（如 0.05 → 0.03）
- ✓ 增加等值线层级

### 场景3：簇太多
- ✓ 增大 `max_buffer_distance`（如 5 → 10）
- ✓ 增大 `min_cluster_size`（如 10 → 20）

### 场景4：簇太少
- ✓ 减小 `max_buffer_distance`（如 5 → 3）
- ✓ 减小 `min_cluster_size`（如 10 → 5）

## 🔧 技术实现

### 后端
- 使用 `ContourBasedClusteringWithoutZ22_map.py`
- 返回等值线多边形坐标数据
- 格式：`[[lon, lat], [lon, lat], ...]`

### 前端
- Leaflet.js 地图库
- `L.polygon()` 绘制等值线边界
- `L.circleMarker()` 绘制客户点
- 自动颜色生成（HSL色彩空间）

### 数据流

```
后端聚类
  ↓
生成等值线多边形
  ↓
转换为坐标数组
  ↓
JSON格式返回前端
  ↓
Leaflet绘制多边形
  ↓
添加交互事件
```

## 📝 API响应格式

```json
{
  "success": true,
  "n_clusters": 13,
  "n_noise": 0,
  "labels": [0, 0, 1, 2, ...],
  "contour_polygons": [
    {
      "cluster_id": 0,
      "polygon": [[112.9, 28.2], [112.91, 28.21], ...],
      "center": [112.95, 28.25],
      "point_count": 487
    },
    ...
  ],
  "metrics": {
    "silhouette_score": 0.5543,
    ...
  }
}
```

## 🐛 常见问题

### Q1: 地图不显示
**原因**: 无法访问 OpenStreetMap 服务器  
**解决**: 
- 检查网络连接
- 尝试使用VPN
- 或切换回百度地图版本

### Q2: 等值线不显示
**原因**: 后端未返回多边形数据  
**解决**:
- 检查浏览器控制台错误
- 确认使用 `ContourBasedClusteringWithoutZ22_map.py`
- 查看服务器日志

### Q3: 颜色都一样
**原因**: 前端颜色映射失败  
**解决**:
- 检查 `labels` 数据是否正确
- 查看浏览器控制台错误
- 刷新页面重试

### Q4: 多边形形状奇怪
**原因**: 等值线算法参数不合适  
**解决**:
- 调整 `grid_size` 参数
- 增大 `max_buffer_distance`
- 尝试不同参数组合

## 🔄 版本切换

### 切换到百度地图版本

修改 `clustering_server.py`:

```python
@app.route('/')
def index():
    return send_from_directory('.', 'interactive_clustering.html')
```

### 切换到 Leaflet 版本（当前）

```python
@app.route('/')
def index():
    return send_from_directory('.', 'interactive_clustering_leaflet.html')
```

## 📈 性能对比

| 指标 | Leaflet版本 | 百度地图版本 |
|------|------------|-------------|
| 地图加载速度 | 较慢（国际网络） | 快（国内网络） |
| 渲染性能 | 优秀 | 优秀 |
| 多边形绘制 | 原生支持 | 需要额外处理 |
| 内存占用 | 相似 | 相似 |

## 🎨 自定义样式

### 修改多边形样式

在 `interactive_clustering_leaflet.html` 中修改：

```javascript
const leafletPolygon = L.polygon(latlngs, {
    color: color,           // 边框颜色
    weight: 2,              // 边框宽度
    opacity: 0.8,           // 边框透明度
    fillColor: color,       // 填充颜色
    fillOpacity: 0.2        // 填充透明度（调整这个值）
}).addTo(map);
```

### 修改客户点样式

```javascript
const marker = L.circleMarker([client.lat, client.lon], {
    radius: 5,              // 圆点半径（调整大小）
    fillColor: color,       // 填充颜色
    color: 'white',         // 边框颜色
    weight: 1,              // 边框宽度
    opacity: 0.8,           // 边框透明度
    fillOpacity: 0.8        // 填充透明度
}).addTo(map);
```

## 📞 技术支持

- 查看 `test_clustering_simple.py` 了解聚类算法测试
- 查看浏览器控制台（F12）查看前端错误
- 查看服务器终端查看后端日志

---

**版本**: v1.1 (Leaflet)  
**更新时间**: 2025-12-31  
**特性**: 等值线边界显示 + 交互式聚类分析
