"""
创建纯Leaflet.js的HTML地图（不使用聚合，所有点直接显示）
"""

import json

# 读取客户数据
print("正在读取客户数据...")
with open('client_data.json', 'r', encoding='utf-8') as f:
    clients = json.load(f)

print(f"✓ 加载了 {len(clients)} 个客户")

# 生成HTML文件
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>湖南省客户分布地图</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 100%;
        }}
        .info-box {{
            position: absolute;
            top: 10px;
            left: 60px;
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
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
        }}
        .legend-item {{
            margin: 5px 0;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-box">
        <h3>湖南省客户分布地图</h3>
        <p>总客户数: <strong>{len(clients)}</strong></p>
        <p>中心: 长沙市</p>
        <p style="font-size: 12px; color: #666;">所有点直接显示，无聚合</p>
    </div>
    
    <div class="legend">
        <h4 style="margin: 0 0 10px 0;">图例</h4>
        <div class="legend-item">
            <span class="legend-color" style="background: #ff4444;"></span>
            重点客户
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #4444ff;"></span>
            普通客户
        </div>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // 初始化地图，中心设置在长沙
        var map = L.map('map').setView([28.2282, 112.9388], 7);
        
        // 添加OpenStreetMap图层
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 18
        }}).addTo(map);
        
        // 客户数据
        var clients = {json.dumps(clients, ensure_ascii=False)};
        
        // 创建图层组（不使用聚合，直接显示所有点）
        var allMarkers = L.layerGroup();
        
        // 添加客户标记
        clients.forEach(function(client) {{
            // 根据客户等级选择颜色
            var color = client.degree === '2' ? '#ff4444' : '#4444ff';
            
            // 使用CircleMarker创建更小的圆点
            var marker = L.circleMarker([client.lat, client.lon], {{
                radius: 3,  // 圆点半径（像素）
                fillColor: color,
                color: 'white',  // 边框颜色
                weight: 1,  // 边框宽度
                opacity: 0.8,
                fillOpacity: 0.6
            }});
            
            // 创建弹出窗口内容
            var popupContent = '<div style="min-width: 200px;">' +
                '<h4 style="margin: 0 0 10px 0;">' + client.name + '</h4>' +
                '<p><b>客户ID:</b> ' + client.id + '</p>' +
                '<p><b>等级:</b> ' + (client.degree === '2' ? '重点客户' : '普通客户') + '</p>' +
                '<p><b>城市:</b> ' + (client.city || '未知') + '</p>' +
                '<p><b>区域:</b> ' + (client.area || '未知') + '</p>' +
                '<p><b>地址:</b> ' + (client.street || '未知') + '</p>' +
                '<p><b>坐标:</b> (' + client.lat.toFixed(6) + ', ' + client.lon.toFixed(6) + ')</p>' +
                '</div>';
            
            marker.bindPopup(popupContent);
            marker.bindTooltip(client.name, {{ direction: 'top', opacity: 0.8 }});
            
            allMarkers.addLayer(marker);
        }});
        
        // 将所有标记添加到地图
        map.addLayer(allMarkers);
        
        // 添加比例尺
        L.control.scale({{ imperial: false }}).addTo(map);
        
        console.log('地图加载完成，共 ' + clients.length + ' 个客户标记（所有点直接显示）');
    </script>
</body>
</html>"""

# 保存HTML文件
output_file = 'client_map_leaflet.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ Leaflet地图已保存到: {output_file}")
print(f"\n地图特性:")
print(f"  - 使用Leaflet.js库")
print(f"  - 中心位置: 长沙 (28.2282, 112.9388)")
print(f"  - 客户标记: {len(clients)}个")
print(f"  - 标记大小: 3像素半径的小圆点")
print(f"  - 显示方式: 所有点直接显示（无聚合）")
print(f"  - 交互功能: 点击查看详情，鼠标悬停显示名称")
print(f"  - 图例: 区分重点客户和普通客户")
print(f"\n请在浏览器中打开 {output_file} 查看地图")
