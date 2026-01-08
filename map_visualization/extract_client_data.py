"""
从Client_t.sql提取客户数据并转换为地图展示格式
"""

import re
import json

input_file = '../compare/dataset/realData/Client_t.sql'
output_json = 'client_data.json'

print(f"正在读取 {input_file}...")

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"✓ 文件读取成功")

# 提取所有INSERT语句（简化的正则表达式）
insert_pattern = r"Insert into BBG\.CLIENT.*?values \((.*?)\);"
inserts = re.findall(insert_pattern, content, re.DOTALL)

print(f"找到 {len(inserts)} 条INSERT语句")

# 解析数据
clients = []
valid_count = 0
invalid_count = 0

for insert_values in inserts:
    try:
        # 分割值（处理引号内的逗号）
        # 使用更简单的方法：按逗号分割，然后处理引号
        values = []
        current_value = ""
        in_quotes = False
        
        for char in insert_values:
            if char == "'" and (not current_value or current_value[-1] != '\\'):
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current_value.strip())
                current_value = ""
                continue
            current_value += char
        
        if current_value:
            values.append(current_value.strip())
        
        # 提取字段（按顺序）
        if len(values) >= 16:
            cli_name = values[0].strip("'")
            cli_degree = values[1].strip("'")
            cli_lat = values[6].strip("'") if values[6] != 'null' else None
            cli_lon = values[7].strip("'") if values[7] != 'null' else None
            cli_province = values[8].strip("'") if values[8] != 'null' else None
            cli_city = values[9].strip("'") if values[9] != 'null' else None
            cli_area = values[10].strip("'") if values[10] != 'null' else None
            cli_street = values[11].strip("'") if values[11] != 'null' else None
            clientid = values[15].strip("'")
            
            # 只保留有经纬度的记录
            if cli_lat and cli_lon:
                try:
                    lat = float(cli_lat)
                    lon = float(cli_lon)
                    
                    # 验证经纬度范围（湖南省大致范围）
                    if 24 <= lat <= 31 and 108 <= lon <= 115:
                        clients.append({
                            'id': int(clientid),
                            'name': cli_name,
                            'degree': cli_degree,
                            'lat': lat,
                            'lon': lon,
                            'province': cli_province,
                            'city': cli_city,
                            'area': cli_area,
                            'street': cli_street
                        })
                        valid_count += 1
                    else:
                        invalid_count += 1
                except:
                    invalid_count += 1
            else:
                invalid_count += 1
    except Exception as e:
        invalid_count += 1
        continue

print(f"\n数据统计:")
print(f"  总记录数: {len(inserts)}")
print(f"  有效记录: {valid_count}")
print(f"  无效记录: {invalid_count}")

# 保存为JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(clients, f, ensure_ascii=False, indent=2)

print(f"\n✓ 数据已保存到: {output_json}")

# 统计城市分布
if clients:
    city_count = {}
    for client in clients:
        city = client.get('city', '未知')
        city_count[city] = city_count.get(city, 0) + 1

    print(f"\n城市分布（前10）:")
    sorted_cities = sorted(city_count.items(), key=lambda x: x[1], reverse=True)
    for city, count in sorted_cities[:10]:
        print(f"  {city}: {count}个客户")
else:
    print("\n警告: 没有提取到有效数据")
    print("请检查SQL文件格式")
