"""
测试聚类服务器API
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_get_clients():
    """测试获取客户数据API"""
    print("\n" + "="*60)
    print("测试1: 获取客户数据")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/clients")
        data = response.json()
        
        if data['success']:
            print(f"✓ 成功获取 {len(data['clients'])} 个客户数据")
            print(f"  示例客户: {data['clients'][0]['name']}")
            return True
        else:
            print(f"✗ 失败: {data.get('error', '未知错误')}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False

def test_clustering():
    """测试聚类分析API"""
    print("\n" + "="*60)
    print("测试2: 执行聚类分析")
    print("="*60)
    
    params = {
        "grid_size": 0.1,  # 使用较大的grid_size加快测试
        "max_buffer_distance": 5,
        "min_cluster_size": 10
    }
    
    print(f"参数: {json.dumps(params, indent=2)}")
    print("正在执行聚类...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/cluster",
            json=params,
            headers={'Content-Type': 'application/json'}
        )
        elapsed_time = time.time() - start_time
        
        data = response.json()
        
        if data['success']:
            print(f"\n✓ 聚类成功 (耗时: {elapsed_time:.2f}秒)")
            print(f"\n聚类结果:")
            print(f"  簇数: {data['n_clusters']}")
            print(f"  噪声点: {data['n_noise']}")
            print(f"  有效点数: {data['n_valid_points']}")
            
            if data.get('metrics'):
                metrics = data['metrics']
                print(f"\n评价指标:")
                print(f"  轮廓系数: {metrics.get('silhouette_score', 'N/A'):.4f}")
                print(f"  CH指数: {metrics.get('calinski_harabasz_score', 'N/A'):.2f}")
                print(f"  DB指数: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
                print(f"  簇内误差: {metrics.get('inertia', 'N/A'):.2f}")
            
            print(f"\n各簇点数:")
            cluster_counts = data.get('cluster_counts', {})
            for cluster_id in sorted(cluster_counts.keys(), key=lambda x: int(x)):
                count = cluster_counts[cluster_id]
                percentage = (count / data['n_valid_points']) * 100
                print(f"  簇 {cluster_id}: {count} 个点 ({percentage:.2f}%)")
            
            return True
        else:
            print(f"✗ 聚类失败: {data.get('error', '未知错误')}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("交互式聚类服务器API测试")
    print("="*60)
    print(f"服务器地址: {BASE_URL}")
    print("="*60)
    
    # 测试1: 获取客户数据
    test1_passed = test_get_clients()
    
    # 测试2: 执行聚类
    test2_passed = test_clustering()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"测试1 (获取客户数据): {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"测试2 (执行聚类分析): {'✓ 通过' if test2_passed else '✗ 失败'}")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("\n✓✓✓ 所有测试通过！系统运行正常。")
        print("\n现在可以打开浏览器访问: http://localhost:5000")
    else:
        print("\n✗✗✗ 部分测试失败，请检查服务器是否正常运行。")

if __name__ == '__main__':
    main()
