"""
创建百度地图版本的客户分布地图
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
    <title>湖南省客户分布地图（百度地图）</title>
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
        <h3>湖南省客户分布地图</h3>
        <p>总客户数: <strong>{len(clients)}</strong></p>
        <p>中心: 长沙市</p>
        <p style="font-size: 12px; color: #666;">使用百度地图</p>
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
    
    <!-- 百度地图API -->
    <script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak=YOUR_BAIDU_MAP_AK"></script>
    
    <script type="text/javascript">
        // 客户数据
        var clients = {json.dumps(clients, ensure_ascii=False)};
        
        // 初始化地图，中心设置在长沙
        var map = new BMap.Map("map");
        var point = new BMap.Point(112.9388, 28.2282);  // 长沙坐标
        map.centerAndZoom(point, 8);
        
        // 启用滚轮缩放
        map.enableScrollWheelZoom(true);
        
        // 添加地图控件
        map.addControl(new BMap.NavigationControl());  // 平移缩放控件
        map.addControl(new BMap.ScaleControl());       // 比例尺控件
        map.addControl(new BMap.OverviewMapControl()); // 缩略地图控件
        
        // 添加客户标记
        var markers = [];
        clients.forEach(function(client) {{
            // 创建点坐标
            var pt = new BMap.Point(client.lon, client.lat);
            
            // 根据客户等级选择颜色
            var color = client.degree === '2' ? '#ff4444' : '#4444ff';
            
            // 创建圆形标记
            var circle = new BMap.Circle(pt, 100, {{
                strokeColor: 'white',
                strokeWeight: 1,
                strokeOpacity: 0.8,
                fillColor: color,
                fillOpacity: 0.6
            }});
            
            map.addOverlay(circle);
            
            // 创建信息窗口内容
            var infoContent = '<div style="padding: 10px; min-width: 250px;">' +
                '<h4 style="margin: 0 0 10px 0; color: #333;">' + client.name + '</h4>' +
                '<p style="margin: 5px 0;"><b>客户ID:</b> ' + client.id + '</p>' +
                '<p style="margin: 5px 0;"><b>等级:</b> ' + (client.degree === '2' ? '<span style="color: red;">重点客户</span>' : '普通客户') + '</p>' +
                '<p style="margin: 5px 0;"><b>城市:</b> ' + (client.city || '未知') + '</p>' +
                '<p style="margin: 5px 0;"><b>区域:</b> ' + (client.area || '未知') + '</p>' +
                '<p style="margin: 5px 0;"><b>地址:</b> ' + (client.street || '未知') + '</p>' +
                '<p style="margin: 5px 0;"><b>坐标:</b> (' + client.lat.toFixed(6) + ', ' + client.lon.toFixed(6) + ')</p>' +
                '</div>';
            
            // 添加点击事件
            circle.addEventListener('click', function() {{
                var infoWindow = new BMap.InfoWindow(infoContent, {{
                    width: 280,
                    height: 200,
                    title: '客户信息'
                }});
                map.openInfoWindow(infoWindow, pt);
            }});
            
            // 添加鼠标悬停事件（显示客户名称）
            circle.addEventListener('mouseover', function() {{
                var label = new BMap.Label(client.name, {{
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
                
                // 保存label引用以便移除
                circle._label = label;
            }});
            
            circle.addEventListener('mouseout', function() {{
                if (circle._label) {{
                    map.removeOverlay(circle._label);
                    circle._label = null;
                }}
            }});
            
            markers.push(circle);
        }});
        
        console.log('百度地图加载完成，共 ' + clients.length + ' 个客户标记');
        
        // 自适应视野（显示所有标记）
        if (clients.length > 0) {{
            var points = clients.map(function(client) {{
                return new BMap.Point(client.lon, client.lat);
            }});
            map.setViewport(points);
        }}
    </script>
</body>
</html>"""

# 保存HTML文件
output_file = 'client_map_baidu.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ 百度地图已保存到: {output_file}")
print(f"\n地图特性:")
print(f"  - 使用百度地图API")
print(f"  - 中心位置: 长沙 (112.9388, 28.2282)")
print(f"  - 客户标记: {len(clients)}个")
print(f"  - 标记样式: 小圆圈（半径100米）")
print(f"  - 显示方式: 所有点直接显示")
print(f"  - 交互功能: 点击查看详情，鼠标悬停显示名称")
print(f"  - 自动适配: 自动调整视野显示所有标记")
print(f"\n注意事项:")
print(f"  1. 需要百度地图API密钥（AK）")
print(f"  2. 请将HTML中的 'YOUR_BAIDU_MAP_AK' 替换为您的密钥")
print(f"  3. 申请地址: https://lbsyun.baidu.com/apiconsole/key")
print(f"\n请在浏览器中打开 {output_file} 查看地图")
