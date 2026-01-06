"""测试城市数据是否可用"""
import json

with open('client_data_by_city/index.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

cities = [c for c in data['cities'] if c['count'] >= 50 and c['city'] is not None]

print(f"符合条件的城市数（客户数>=50）: {len(cities)}")
print("\n城市列表:")
for city in cities:
    print(f"  {city['city']}: {city['count']}个客户")
