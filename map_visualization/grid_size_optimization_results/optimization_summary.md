# grid_size优化结果汇总报告

## 参数设置

- grid_size范围: 6 - 7
- 步长: 0.001
- max_buffer_distance: 0
- min_cluster_size: 5
- 最少点数: 50

## 优化结果

成功优化的城市数: 1

| 城市 | 客户数 | 最佳grid_size | 轮廓系数 | 簇数 | 噪声点 |
|------|--------|---------------|----------|------|--------|
| 长沙市 | 3229 | 6.012 | 0.6177 | 2 | 0 |

## 统计分析

- 最佳grid_size平均值: 6.012
- 最佳grid_size中位数: 6.012
- 最佳grid_size标准差: 0.000
- 最佳grid_size范围: [6.012, 6.012]

- 轮廓系数平均值: 0.6177
- 轮廓系数中位数: 0.6177
- 轮廓系数范围: [0.6177, 0.6177]

## 推荐参数

基于所有城市的优化结果，推荐使用:

- **grid_size**: 6.012
- max_buffer_distance: 0
- min_cluster_size: 5

- **grid_size**: 6.012
- max_buffer_distance: 0
- min_cluster_size: 5

