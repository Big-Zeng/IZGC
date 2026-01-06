# 客户地图可视化与聚类分析系统

本目录包含客户数据的地图可视化和交互式聚类分析工具。

## 🎯 主要功能

### 1. 静态地图可视化
- 在百度地图上显示所有客户位置
- 查看客户详细信息

### 2. 交互式聚类分析 ⭐ **新功能**
- 在浏览器中动态调整聚类参数
- 实时查看聚类结果和评价指标
- 彩色地图显示聚类结果
- 支持多种评价指标

## 🚀 快速开始

### 交互式聚类分析（推荐）

1. **安装依赖**:
```bash
pip install flask flask-cors numpy scipy scikit-learn shapely matplotlib
```

2. **启动服务器**:
```bash
# Windows用户
双击运行: 启动交互式聚类服务.bat

# 或手动启动
python clustering_server.py
```

3. **访问系统**:
打开浏览器访问: http://localhost:5000

### 静态地图查看

1. **启动本地服务器**:
```bash
python start_server.py
```

2. **访问地图**:
打开浏览器访问: http://localhost:8000/client_map_baidu.html

## 📁 文件说明

### 交互式聚类系统
- `clustering_server.py` - Flask后端服务器
- `interactive_clustering.html` - 前端交互页面
- `启动交互式聚类服务.bat` - Windows启动脚本
- `test_server.py` - API测试脚本
- `快速开始.md` - 快速启动指南
- `交互式聚类使用说明.md` - 详细使用说明
- `系统说明.md` - 系统架构说明

### 静态地图
- `client_map_baidu.html` - 百度地图版本
- `client_map_leaflet.html` - Leaflet地图版本
- `start_server.py` - 本地HTTP服务器
- `启动地图服务器.bat` - Windows启动脚本

### 聚类分析
- `cluster_analysis.py` - 命令行聚类分析脚本
- `create_clustered_map.py` - 生成聚类结果地图
- `clustering_metrics.json` - 聚类评价指标
- `client_data_clustered.json` - 带聚类标签的客户数据
- `聚类分析结果报告.md` - 聚类结果报告

### 数据处理
- `extract_client_data_simple.py` - 从SQL提取客户数据
- `client_data.json` - 客户数据（8,489个点）

## 📊 系统特点

### 交互式聚类分析
- ✓ 实时参数调整
- ✓ 动态聚类计算
- ✓ 可视化结果展示
- ✓ 多种评价指标
- ✓ 无需重启服务器

### 聚类算法
- 等值线聚类 (ContourBasedClusteringWithoutZ22)
- 基于密度的空间聚类
- 自动小簇合并
- 确保无噪声点

### 评价指标
- 轮廓系数 (Silhouette Score)
- Calinski-Harabasz 指数
- Davies-Bouldin 指数
- 簇内误差平方和

## 📖 使用文档

- **快速开始**: 查看 `快速开始.md`
- **详细说明**: 查看 `交互式聚类使用说明.md`
- **系统架构**: 查看 `系统说明.md`
- **聚类结果**: 查看 `聚类分析结果报告.md`

## 🎨 界面预览

### 交互式聚类界面
- 左侧: 参数调整面板 + 结果显示
- 右侧: 百度地图 + 聚类可视化
- 实时: 点击聚类 → 查看结果 → 调整参数

### 功能特性
- 滑块调整参数
- 一键执行聚类
- 彩色簇显示
- 点击查看详情
- 图例说明

## 🔧 技术栈

- **后端**: Python + Flask + Flask-CORS
- **前端**: HTML5 + CSS3 + JavaScript
- **地图**: 百度地图 JavaScript API v3.0
- **算法**: ContourBasedClusteringWithoutZ22
- **评价**: Scikit-learn + SciPy + NumPy

## 📈 性能

- 数据量: 8,489 个客户点
- 计算时间: 5-30 秒（取决于参数）
- 内存占用: 200-500 MB

## 🐛 常见问题

**Q: 服务器启动失败？**  
A: 运行 `pip install flask flask-cors`

**Q: 地图不显示？**  
A: 检查百度地图AK配置

**Q: 聚类太慢？**  
A: 增大 grid_size 参数

更多问题请查看 `交互式聚类使用说明.md`

## 📞 测试系统

运行测试脚本验证系统是否正常：
```bash
python test_server.py
```

---

**更新时间**: 2025-12-31  
**版本**: v1.0  
**功能**: 静态地图 + 交互式聚类分析
