"""
按城市和区域分割客户数据
将客户数据按照城市（city）和区域（area）分组，保存为不同的数据集
"""

import json
import os
from collections import defaultdict

print("="*80)
print("按城市和区域分割客户数据")
print("="*80)

# 读取客户数据
print("\n正在读取客户数据...")
with open('client_data.json', 'r', encoding='utf-8') as f:
    clients = json.load(f)

print(f"✓ 加载了 {len(clients)} 个客户")

# 创建输出目录
output_dir = 'client_data_by_city'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ 创建输出目录: {output_dir}")

# 按城市分组
city_groups = defaultdict(list)
city_area_groups = defaultdict(lambda: defaultdict(list))

for client in clients:
    city = client.get('city', '未知城市')
    area = client.get('area', '未知区域')
    
    # 按城市分组
    city_groups[city].append(client)
    
    # 按城市+区域分组
    city_area_groups[city][area].append(client)

# 统计信息
print(f"\n发现 {len(city_groups)} 个城市")
print(f"发现 {sum(len(areas) for areas in city_area_groups.values())} 个城市-区域组合")

# 保存按城市分组的数据
print("\n" + "="*80)
print("保存按城市分组的数据")
print("="*80)

city_stats = []
for city, city_clients in sorted(city_groups.items(), key=lambda x: len(x[1]), reverse=True):
    # 保存JSON文件
    filename = f"{output_dir}/{city}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(city_clients, f, ensure_ascii=False, indent=2)
    
    # 保存CSV文件（用于聚类算法）
    csv_filename = f"{output_dir}/{city}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("lon,lat,id,name,degree,city,area,street\n")
        # 写入数据
        for client in city_clients:
            f.write(f"{client['lon']},{client['lat']},{client['id']},{client['name']},"
                   f"{client.get('degree', '')},"
                   f"{client.get('city', '')},"
                   f"{client.get('area', '')},"
                   f"\"{client.get('street', '')}\"\n")
    
    city_stats.append({
        'city': city,
        'count': len(city_clients),
        'json_file': filename,
        'csv_file': csv_filename
    })
    
    print(f"✓ {city}: {len(city_clients)} 个客户")
    print(f"  - JSON: {filename}")
    print(f"  - CSV:  {csv_filename}")

# 保存按城市+区域分组的数据
print("\n" + "="*80)
print("保存按城市+区域分组的数据")
print("="*80)

# 创建城市-区域子目录
city_area_dir = f"{output_dir}/by_area"
if not os.path.exists(city_area_dir):
    os.makedirs(city_area_dir)

area_stats = []
for city in sorted(city_area_groups.keys(), key=lambda x: (x is None, x)):
    areas = city_area_groups[city]
    print(f"\n城市: {city if city else '未知城市'}")
    
    for area, area_clients in sorted(areas.items(), key=lambda x: len(x[1]), reverse=True):
        # 创建安全的文件名
        safe_city = (city if city else '未知城市').replace('/', '_').replace('\\', '_')
        safe_area = (area if area else '未知区域').replace('/', '_').replace('\\', '_')
        
        # 保存JSON文件
        filename = f"{city_area_dir}/{safe_city}_{safe_area}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(area_clients, f, ensure_ascii=False, indent=2)
        
        # 保存CSV文件
        csv_filename = f"{city_area_dir}/{safe_city}_{safe_area}.csv"
        with open(csv_filename, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("lon,lat,id,name,degree,city,area,street\n")
            # 写入数据
            for client in area_clients:
                f.write(f"{client['lon']},{client['lat']},{client['id']},{client['name']},"
                       f"{client.get('degree', '')},"
                       f"{client.get('city', '')},"
                       f"{client.get('area', '')},"
                       f"\"{client.get('street', '')}\"\n")
        
        area_stats.append({
            'city': city,
            'area': area,
            'count': len(area_clients),
            'json_file': filename,
            'csv_file': csv_filename
        })
        
        print(f"  ✓ {area}: {len(area_clients)} 个客户")

# 生成统计报告
print("\n" + "="*80)
print("生成统计报告")
print("="*80)

report_lines = []
report_lines.append("# 客户数据分组统计报告\n")
report_lines.append(f"\n总客户数: {len(clients)}\n")
report_lines.append(f"城市数量: {len(city_groups)}\n")
report_lines.append(f"城市-区域组合数: {len(area_stats)}\n")

report_lines.append("\n## 按城市分组统计\n")
report_lines.append("\n| 城市 | 客户数 | 占比 | JSON文件 | CSV文件 |\n")
report_lines.append("|------|--------|------|----------|----------|\n")

for stat in sorted(city_stats, key=lambda x: x['count'], reverse=True):
    percentage = (stat['count'] / len(clients)) * 100
    report_lines.append(f"| {stat['city']} | {stat['count']} | {percentage:.2f}% | "
                       f"{stat['json_file']} | {stat['csv_file']} |\n")

report_lines.append("\n## 按城市+区域分组统计（前20）\n")
report_lines.append("\n| 城市 | 区域 | 客户数 | 占比 | JSON文件 |\n")
report_lines.append("|------|------|--------|------|----------|\n")

for stat in sorted(area_stats, key=lambda x: x['count'], reverse=True)[:20]:
    percentage = (stat['count'] / len(clients)) * 100
    report_lines.append(f"| {stat['city']} | {stat['area']} | {stat['count']} | "
                       f"{percentage:.2f}% | {stat['json_file']} |\n")

# 保存报告
report_file = f"{output_dir}/分组统计报告.md"
with open(report_file, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"✓ 统计报告已保存: {report_file}")

# 生成索引文件
print("\n生成索引文件...")
index_data = {
    'total_clients': len(clients),
    'total_cities': len(city_groups),
    'total_areas': len(area_stats),
    'cities': city_stats,
    'city_areas': area_stats
}

index_file = f"{output_dir}/index.json"
with open(index_file, 'w', encoding='utf-8') as f:
    json.dump(index_data, f, ensure_ascii=False, indent=2)

print(f"✓ 索引文件已保存: {index_file}")

# 打印摘要
print("\n" + "="*80)
print("分组完成摘要")
print("="*80)
print(f"\n总客户数: {len(clients)}")
print(f"城市数量: {len(city_groups)}")
print(f"城市-区域组合数: {len(area_stats)}")

print(f"\n前10个城市（按客户数排序）:")
for i, stat in enumerate(sorted(city_stats, key=lambda x: x['count'], reverse=True)[:10], 1):
    percentage = (stat['count'] / len(clients)) * 100
    print(f"  {i}. {stat['city']}: {stat['count']} 个客户 ({percentage:.2f}%)")

print(f"\n输出目录: {output_dir}/")
print(f"  - 按城市: {len(city_stats)} 个文件")
print(f"  - 按区域: {len(area_stats)} 个文件")
print(f"  - 统计报告: {report_file}")
print(f"  - 索引文件: {index_file}")

print("\n" + "="*80)
print("✓ 数据分组完成")
print("="*80)
