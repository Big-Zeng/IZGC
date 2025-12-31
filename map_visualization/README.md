# 客户分布地图可视化

将Client_t.sql中的客户数据在地图上可视化展示。

## 文件说明

- `extract_client_data.py` - 从SQL文件提取客户数据并保存为JSON
- `create_leaflet_map.py` - 创建纯Leaflet.js的HTML地图（推荐）
- `create_map.py` - 使用Folium库创建地图（需要安装folium）
- `client_data.json` - 提取的客户数据（运行后生成）
- `client_map_leaflet.html` - Leaflet地图文件（运行后生成）
- `client_map.html` - Folium地图文件（运行后生成）

## 使用步骤

### 方法1: 使用纯Leaflet.js（推荐，无需额外安装）

```bash
# 1. 提取数据
cd map_visualization
python extract_client_data.py

# 2. 创建Leaflet地图
python create_leaflet_map.py

# 3. 在浏览器中打开
# 双击 client_map_leaflet.html 或在浏览器中打开
```

### 方法2: 使用Folium库（需要安装）

```bash
# 1. 安装依赖
pip install folium

# 2. 提取数据
cd map_visualization
python extract_client_data.py

# 3. 创建Folium地图
python create_map.py

# 4. 在浏览器中打开
# 双击 client_map.html 或在浏览器中打开
```

## 地图特性

### Leaflet地图特性
- ✅ 中心位置：湖南长沙 (28.2282, 112.9388)
- ✅ 标记聚合：密集区域自动聚合，点击展开
- ✅ 客户分类：红色=重点客户，蓝色=普通客户
- ✅ 交互功能：
  - 点击标记查看客户详细信息
  - 鼠标悬停显示客户名称
  - 缩放和平移地图
- ✅ 信息面板：显示总客户数
- ✅ 图例：颜色说明
- ✅ 比例尺：显示距离

### Folium地图特性（额外功能）
- ✅ 多图层：聚合标记、热力图、城市分组
- ✅ 图层控制：可切换不同图层
- ✅ 全屏模式：点击按钮全屏查看
- ✅ 鼠标位置：实时显示经纬度

## 数据说明

### 客户数据字段
- `id`: 客户编码
- `name`: 客户名称
- `degree`: 客户等级（1=普通，2=重点）
- `lat`: 纬度
- `lon`: 经度
- `province`: 省份
- `city`: 城市
- `area`: 区域
- `street`: 街道地址

### 数据过滤
- 只保留有经纬度的记录
- 只保留湖南省范围内的坐标（纬度24-31，经度108-115）

## 技术栈

- **Leaflet.js**: 开源JavaScript地图库
- **Leaflet.markercluster**: 标记聚合插件
- **OpenStreetMap**: 免费地图瓦片服务
- **Python**: 数据提取和处理
- **Folium**: Python的Leaflet封装库（可选）

## 注意事项

1. 地图需要联网才能加载地图瓦片
2. 如果标记过多，建议使用聚合功能
3. 可以根据需要调整地图中心和缩放级别
4. 客户数据包含真实地址信息，注意数据安全

## 自定义

### 修改地图中心
在脚本中修改：
```python
map_center = [28.2282, 112.9388]  # [纬度, 经度]
```

### 修改缩放级别
```python
zoom_start=7  # 1-18，数字越大越详细
```

### 修改标记颜色
在脚本中修改颜色判断逻辑：
```python
color = 'red' if client['degree'] == '2' else 'blue'
```

## 示例截图

地图将显示：
- 湖南省地图背景
- 客户位置标记（聚合显示）
- 点击标记显示客户详情
- 信息面板和图例

## 问题排查

### 地图不显示
- 检查网络连接（需要加载地图瓦片）
- 检查浏览器控制台是否有错误
- 确认client_data.json文件存在

### 标记不显示
- 检查客户数据是否有效
- 确认经纬度范围正确
- 查看浏览器控制台日志

### 性能问题
- 如果客户数量过多（>10000），考虑：
  - 使用标记聚合（已启用）
  - 分批加载数据
  - 使用热力图代替标记
