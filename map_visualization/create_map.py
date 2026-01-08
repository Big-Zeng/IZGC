"""
使用Folium创建交互式地图展示客户分布
Folium是Python的Leaflet封装库
"""

import json
import folium
from folium.plugins import MarkerCluster, HeatMap

# 读取客户数据
print("正在读取客户数据...")
with open('client_data.json', 'r', encoding='utf-8') as f:
    clients = json.load(f)

print(f"✓ 加载了 {len(clients)} 个客户")

# 创建地图，中心设置在湖南长沙
# 长沙坐标: 28.2282, 112.9388
map_center = [28.2282, 112.9388]
m = folium.Map(
    location=map_center,
    zoom_start=7,
    tiles='OpenStreetMap'
)

# 添加标题
title_html = '''
<div style="position: fixed; 
            top: 10px; left: 50px; width: 300px; height: 60px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:16px; padding: 10px">
<b>湖南省客户分布地图</b><br>
总客户数: {count}
</div>
'''.format(count=len(clients))
m.get_root().html.add_child(folium.Element(title_html))

# 方法1: 使用MarkerCluster（聚合标记）
print("创建聚合标记图层...")
marker_cluster = MarkerCluster(name='客户标记（聚合）').add_to(m)

for client in clients:
    # 创建弹出窗口内容
    popup_html = f"""
    <div style="width: 200px;">
        <h4>{client['name']}</h4>
        <p><b>客户ID:</b> {client['id']}</p>
        <p><b>等级:</b> {'重点客户' if client['degree'] == '2' else '普通客户'}</p>
        <p><b>城市:</b> {client['city']}</p>
        <p><b>区域:</b> {client['area']}</p>
        <p><b>地址:</b> {client['street']}</p>
        <p><b>坐标:</b> ({client['lat']:.6f}, {client['lon']:.6f})</p>
    </div>
    """
    
    # 根据客户等级设置颜色
    color = 'red' if client['degree'] == '2' else 'blue'
    
    folium.Marker(
        location=[client['lat'], client['lon']],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=client['name'],
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(marker_cluster)

# 方法2: 创建热力图图层
print("创建热力图图层...")
heat_data = [[client['lat'], client['lon']] for client in clients]
HeatMap(
    heat_data,
    name='客户密度热力图',
    min_opacity=0.3,
    radius=15,
    blur=20,
    max_zoom=13
).add_to(m)

# 方法3: 按城市分组的图层
print("创建城市分组图层...")
city_groups = {}
for client in clients:
    city = client.get('city', '未知')
    if city not in city_groups:
        city_groups[city] = folium.FeatureGroup(name=f'{city}客户', show=False)
        city_groups[city].add_to(m)
    
    folium.CircleMarker(
        location=[client['lat'], client['lon']],
        radius=5,
        popup=f"{client['name']}<br>{client['street']}",
        tooltip=client['name'],
        color='blue',
        fill=True,
        fillColor='blue',
        fillOpacity=0.6
    ).add_to(city_groups[city])

# 添加图层控制
folium.LayerControl().add_to(m)

# 添加全屏按钮
from folium.plugins import Fullscreen
Fullscreen().add_to(m)

# 添加鼠标位置显示
from folium.plugins import MousePosition
MousePosition().add_to(m)

# 保存地图
output_file = 'client_map.html'
m.save(output_file)

print(f"\n✓ 地图已保存到: {output_file}")
print(f"\n地图特性:")
print(f"  - 中心位置: 长沙 ({map_center[0]}, {map_center[1]})")
print(f"  - 客户标记: {len(clients)}个")
print(f"  - 图层: 聚合标记、热力图、城市分组")
print(f"  - 交互: 点击标记查看详情，切换图层")
print(f"\n请在浏览器中打开 {output_file} 查看地图")
