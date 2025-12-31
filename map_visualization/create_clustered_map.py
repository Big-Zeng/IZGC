"""
创建显示聚类结果的百度地图
"""

import json
import numpy as np

# 读取聚类后的客户数据
print("正在读取聚类结果...")
with open('client_data_clustered.json', 'r', encoding='utf-8') as f:
    clients = json.load(f)

print(f"✓ 加载了 {len(clients)} 个客户")

# 统计聚类信息
clusters = {}
for client in clients:
    cluster_id = client['cluster']
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(client)

n_clusters = len([c for c in clusters.keys() if c != -1])
n_noise = len(clusters.get(-1, []))

print(f"  簇数: {n_clusters}")
print(f"  噪声点: {n_noise}")

# 为每个簇生成颜色
def generate_colors(n):
    """生成n个不同的颜色"""
    colors = []
    for i in range(n):
        hue = (i * 360 / n) % 360
        colors.append(f'hsl({hue}, 70%, 50%)')
    return colors

cluster_colors = {}
valid_clusters = [c for c in clusters.keys() if c != -1]
colors = generate_colors(len(valid_clusters))

for i, cluster_id in enumerate(sorted(valid_clusters)):
    cluster_colors[cluster_id] = colors[i]

cluster_colors[-1] = '#888888'  # 噪声点用灰色

# 生成HTML
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>客户聚类分析结果（百度地图）</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: "Microsoft YaHei", Arial, sans-serif;
        }}
        #map {{
            width: 100%;
            height: 100%;
        }}
        .info-box {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            max-width: 300px;
        }}
        .info-box h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            max-height: 400px;
            overflow-y: auto;
        }}
        .legend-item {{
            margin: 5px 0;
            font-size: 12px;
        }}
        .legend-color {{
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 50%;
            border: 1px solid white;
        }}
        .BMap_cpyCtrl {{
            display: none !important;
        }}
        .anchorBL {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-box">
        <h3>客户聚类分析结果</h3>
        <p><b>总客户数:</b> {len(clients)}</p>
        <p><b>发现簇数:</b> {n_clusters}</p>
        <p><b>噪声点数:</b> {n_noise}</p>
        <p style="font-size: 12px; color: #666; margin-top: 10px;">
            使用等值线聚类算法<br>
            点击标记查看详情
        </p>
    </div>
    
    <div class="legend">
        <h4 style="margin: 0 0 10px 0;">聚类图例</h4>
"""

# 添加图例项
for cluster_id in sorted(valid_clusters):
    count = len(clusters[cluster_id])
    color = cluster_colors[cluster_id]
    html_content += f"""        <div class="legend-item">
            <span class="legend-color" style="background: {color};"></span>
            簇 {cluster_id} ({count}个客户)
        </div>
"""

if n_noise > 0:
    html_content += f"""        <div class="legend-item">
            <span class="legend-color" style="background: #888888;"></span>
            噪声点 ({n_noise}个)
        </div>
"""

html_content += f"""    </div>
    
    <!-- 百度地图API -->
    <script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak=YOUR_BAIDU_MAP_AK"></script>
    
    <script type="text/javascript">
        // 客户数据
        var clients = {json.dumps(clients, ensure_ascii=False)};
        
        // 簇颜色映射
        var clusterColors = {json.dumps(cluster_colors, ensure_ascii=False)};
        
        // 初始化地图
        var map = new BMap.Map("map");
        var point = new BMap.Point(112.9388, 28.2282);
        map.centerAndZoom(point, 8);
        map.enableScrollWheelZoom(true);
        
        // 添加控件
        map.addControl(new BMap.NavigationControl());
        map.addControl(new BMap.ScaleControl());
        
        // 添加客户标记
        clients.forEach(function(client) {{
            var pt = new BMap.Point(client.lon, client.lat);
            var color = clusterColors[client.cluster] || '#888888';
            
            // 创建圆形标记
            var circle = new BMap.Circle(pt, 100, {{
                strokeColor: 'white',
                strokeWeight: 1,
                strokeOpacity: 0.8,
                fillColor: color,
                fillOpacity: 0.7
            }});
            
            map.addOverlay(circle);
            
            // 信息窗口
            var clusterLabel = client.cluster === -1 ? '噪声点' : '簇 ' + client.cluster;
            var infoContent = '<div style="padding: 10px; min-width: 250px;">' +
                '<h4 style="margin: 0 0 10px 0; color: #333;">' + client.name + '</h4>' +
                '<p style="margin: 5px 0;"><b>客户ID:</b> ' + client.id + '</p>' +
                '<p style="margin: 5px 0;"><b>聚类:</b> <span style="color: ' + color + '; font-weight: bold;">' + clusterLabel + '</span></p>' +
                '<p style="margin: 5px 0;"><b>等级:</b> ' + (client.degree === '2' ? '<span style="color: red;">重点客户</span>' : '普通客户') + '</p>' +
                '<p style="margin: 5px 0;"><b>城市:</b> ' + (client.city || '未知') + '</p>' +
                '<p style="margin: 5px 0;"><b>区域:</b> ' + (client.area || '未知') + '</p>' +
                '<p style="margin: 5px 0;"><b>地址:</b> ' + (client.street || '未知') + '</p>' +
                '</div>';
            
            circle.addEventListener('click', function() {{
                var infoWindow = new BMap.InfoWindow(infoContent, {{
                    width: 280,
                    height: 220,
                    title: '客户信息'
                }});
                map.openInfoWindow(infoWindow, pt);
            }});
            
            // 鼠标悬停
            circle.addEventListener('mouseover', function() {{
                var label = new BMap.Label(client.name + ' (' + clusterLabel + ')', {{
                    position: pt,
                    offset: new BMap.Size(10, -10)
                }});
                label.setStyle({{
                    color: '#333',
                    fontSize: '12px',
                    backgroundColor: 'white',
                    border: '1px solid #ccc',
                    padding: '5px',
                    borderRadius: '3px',
                    boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                }});
                map.addOverlay(label);
                circle._label = label;
            }});
            
            circle.addEventListener('mouseout', function() {{
                if (circle._label) {{
                    map.removeOverlay(circle._label);
                    circle._label = null;
                }}
            }});
        }});
        
        console.log('聚类地图加载完成');
        console.log('簇数: {n_clusters}, 噪声点: {n_noise}');
        
        // 自适应视野
        if (clients.length > 0) {{
            var points = clients.map(function(client) {{
                return new BMap.Point(client.lon, client.lat);
            }});
            map.setViewport(points);
        }}
    </script>
</body>
</html>"""

# 保存HTML
output_file = 'client_map_clustered.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ 聚类地图已保存到: {output_file}")
print(f"\n地图特性:")
print(f"  - 显示 {n_clusters} 个聚类簇")
print(f"  - 每个簇使用不同颜色")
print(f"  - 包含图例说明")
print(f"  - 点击查看客户详情和所属簇")
print(f"\n请使用本地服务器打开:")
print(f"  python start_server.py")
print(f"  然后访问: http://localhost:8000/{output_file}")
