"""
启动支持零值点的交互式聚类服务器
"""

import os
import sys

# 切换到正确的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入并运行服务器
from clustering_server_with_zero_points import app

if __name__ == '__main__':
    print("\n" + "="*80)
    print("交互式聚类分析服务器（支持零值点）")
    print("="*80)
    print("\n功能说明:")
    print("  1. 加载真实数据（长沙市等城市数据）")
    print("  2. 在地图上手动点击添加零值点")
    print("  3. 保存零值点到CSV文件")
    print("  4. 结合零值点执行聚类")
    print("  5. 在地图上显示聚类结果和零值点")
    print("\n使用方法:")
    print("  1. 选择数据集并点击'加载数据'")
    print("  2. 点击'切换到添加零值点模式'")
    print("  3. 在地图上点击添加零值点（绿色标记）")
    print("  4. 点击'保存'将零值点保存到CSV文件")
    print("  5. 调整聚类参数，点击'开始聚类'")
    print("\n" + "="*80)
    print("服务器启动中...")
    print("访问地址: http://localhost:5001")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
