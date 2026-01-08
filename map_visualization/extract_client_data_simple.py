"""
从Client_t.sql提取客户数据（简化版本）
"""

import re
import json

input_file = '../compare/dataset/realData/Client_t.sql'
output_json = 'client_data.json'

print(f"正在读取 {input_file}...")

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"✓ 文件读取成功，共{len(lines)}行")

# 解析数据
clients = []
valid_count = 0
invalid_count = 0

for line in lines:
    if line.strip().startswith('Insert into BBG.CLIENT'):
        try:
            # 提取values部分
            match = re.search(r"values \((.*?)\);", line)
            if not match:
                continue
            
            values_str = match.group(1)
            
            # 手动解析值（处理引号和逗号）
            values = []
            current = ""
            in_quote = False
            paren_depth = 0
            
            for char in values_str:
                if char == "'" and (not current or current[-1] != '\\'):
                    in_quote = not in_quote
                    current += char
                elif char == '(' and not in_quote:
                    paren_depth += 1
                    current += char
                elif char == ')' and not in_quote:
                    paren_depth -= 1
                    current += char
                elif char == ',' and not in_quote and paren_depth == 0:
                    values.append(current.strip())
                    current = ""
                else:
                    current += char
            
            if current:
                values.append(current.strip())
            
            # 提取字段
            if len(values) >= 16:
                cli_name = values[0].strip("'")
                cli_degree = values[1].strip("'")
                cli_lat_str = values[6]
                cli_lon_str = values[7]
                cli_province = values[8].strip("'") if values[8] != 'null' else None
                cli_city = values[9].strip("'") if values[9] != 'null' else None
                cli_area = values[10].strip("'") if values[10] != 'null' else None
                cli_street = values[11].strip("'") if values[11] != 'null' else None
                clientid = values[15]
                
                # 提取经纬度
                if cli_lat_str != 'null' and cli_lon_str != 'null':
                    cli_lat = cli_lat_str.strip("'")
                    cli_lon = cli_lon_str.strip("'")
                    
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
                    except ValueError:
                        invalid_count += 1
                else:
                    invalid_count += 1
        except Exception as e:
            invalid_count += 1
            continue

print(f"\n数据统计:")
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
    
    # 显示前5个客户示例
    print(f"\n前5个客户示例:")
    for i, client in enumerate(clients[:5], 1):
        print(f"  {i}. {client['name']} - {client['city']} ({client['lat']}, {client['lon']})")
else:
    print("\n警告: 没有提取到有效数据")
