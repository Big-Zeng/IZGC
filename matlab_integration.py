"""
Python调用MATLAB代码的集成模块
支持三种方法：
1. MATLAB Engine API (推荐，需要安装MATLAB)
2. Oct2Py (使用Octave，免费替代方案)
3. 子进程调用 (通用方法)
"""

import numpy as np
import os
import subprocess
import tempfile


class MatlabRunner:
    """MATLAB代码运行器"""
    
    def __init__(self, method='auto'):
        """
        初始化MATLAB运行器
        
        参数:
            method: 'engine' (MATLAB Engine), 'octave' (Oct2Py), 'subprocess' (子进程), 'auto' (自动选择)
        """
        self.method = method
        self.matlab_engine = None
        self.octave_engine = None
        
        if method == 'auto':
            self.method = self._detect_available_method()
        
        self._initialize_engine()
    
    def _detect_available_method(self):
        """自动检测可用的方法"""
        # 尝试MATLAB Engine
        try:
            import matlab.engine
            print("检测到 MATLAB Engine API")
            return 'engine'
        except ImportError:
            pass
        
        # 尝试Oct2Py
        try:
            import oct2py
            print("检测到 Oct2Py (Octave)")
            return 'octave'
        except ImportError:
            pass
        
        # 回退到子进程
        print("使用子进程方法调用MATLAB")
        return 'subprocess'
    
    def _initialize_engine(self):
        """初始化引擎"""
        if self.method == 'engine':
            try:
                import matlab.engine
                print("启动 MATLAB Engine...")
                self.matlab_engine = matlab.engine.start_matlab()
                print("MATLAB Engine 启动成功")
            except Exception as e:
                print(f"MATLAB Engine 启动失败: {e}")
                self.method = 'subprocess'
        
        elif self.method == 'octave':
            try:
                from oct2py import Oct2Py
                print("启动 Octave...")
                self.octave_engine = Oct2Py()
                print("Octave 启动成功")
            except Exception as e:
                print(f"Octave 启动失败: {e}")
                self.method = 'subprocess'
    
    def run_algorithm_pr(self, data, n_clusters, k=11, dataset_num=2):
        """
        运行MATLAB的Algorithm_PR算法
        
        参数:
            data: numpy数组，shape=(n_samples, 3)，列为[x, y, label]
            n_clusters: 簇数量
            k: 近邻数量
            dataset_num: 数据集编号（用于MATLAB的switch语句）
            
        返回:
            labels: 聚类标签
            metrics: 评估指标字典 {'AMI': ..., 'ARI': ..., 'FMI': ...}
        """
        if self.method == 'engine':
            return self._run_with_engine(data, n_clusters, k)
        elif self.method == 'octave':
            return self._run_with_octave(data, n_clusters, k)
        else:
            return self._run_with_subprocess(data, n_clusters, k)
    
    def _run_with_engine(self, data, n_clusters, k):
        """使用MATLAB Engine运行"""
        import matlab
        
        # 转换数据为MATLAB格式
        matlab_data = matlab.double(data.tolist())
        
        # 添加MATLAB路径
        self.matlab_engine.addpath('compare/algorithm', nargout=0)
        
        # 调用MATLAB函数
        try:
            # 这里需要创建一个简化的MATLAB函数来调用Algorithm_PR
            labels, ami, ari, fmi = self.matlab_engine.run_algorithm_pr_wrapper(
                matlab_data, float(n_clusters), float(k), nargout=4
            )
            
            # 转换回Python格式
            labels = np.array(labels).flatten()
            
            return labels, {
                'AMI': float(ami),
                'ARI': float(ari),
                'FMI': float(fmi)
            }
        except Exception as e:
            print(f"MATLAB执行错误: {e}")
            return None, None
    
    def _run_with_octave(self, data, n_clusters, k):
        """使用Octave运行"""
        # 添加路径
        self.octave_engine.addpath('compare/algorithm')
        
        try:
            # 调用MATLAB函数
            result = self.octave_engine.run_algorithm_pr_wrapper(
                data, n_clusters, k, nout=4
            )
            
            labels, ami, ari, fmi = result
            
            return labels.flatten(), {
                'AMI': float(ami),
                'ARI': float(ari),
                'FMI': float(fmi)
            }
        except Exception as e:
            print(f"Octave执行错误: {e}")
            return None, None
    
    def _run_with_subprocess(self, data, n_clusters, k):
        """使用子进程运行MATLAB"""
        # 创建临时文件保存数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_data_file = f.name
            np.savetxt(f, data, delimiter=',')
        
        # 创建临时MATLAB脚本
        matlab_script = f"""
        % 临时脚本
        data = readmatrix('{temp_data_file.replace(os.sep, '/')}');
        
        % 添加路径
        addpath('compare/algorithm');
        
        % 运行算法
        [labels, ami, ari, fmi] = run_algorithm_pr_wrapper(data, {n_clusters}, {k});
        
        % 保存结果
        save('temp_matlab_result.mat', 'labels', 'ami', 'ari', 'fmi');
        
        % 退出
        quit;
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as f:
            temp_script_file = f.name
            f.write(matlab_script)
        
        try:
            # 运行MATLAB
            subprocess.run(['matlab', '-batch', f"run('{temp_script_file}')"], 
                         check=True, capture_output=True)
            
            # 读取结果
            from scipy.io import loadmat
            result = loadmat('temp_matlab_result.mat')
            
            labels = result['labels'].flatten()
            ami = float(result['ami'])
            ari = float(result['ari'])
            fmi = float(result['fmi'])
            
            # 清理临时文件
            os.remove(temp_data_file)
            os.remove(temp_script_file)
            os.remove('temp_matlab_result.mat')
            
            return labels, {'AMI': ami, 'ARI': ari, 'FMI': fmi}
            
        except Exception as e:
            print(f"子进程执行错误: {e}")
            # 清理临时文件
            if os.path.exists(temp_data_file):
                os.remove(temp_data_file)
            if os.path.exists(temp_script_file):
                os.remove(temp_script_file)
            return None, None
    
    def close(self):
        """关闭引擎"""
        if self.matlab_engine:
            self.matlab_engine.quit()
        if self.octave_engine:
            self.octave_engine.exit()


# 简化的调用接口
def run_matlab_algorithm_pr(data, n_clusters, k=11):
    """
    简化的接口，运行MATLAB的Algorithm_PR算法
    
    参数:
        data: numpy数组，shape=(n_samples, 3)，列为[x, y, label]
        n_clusters: 簇数量
        k: 近邻数量
        
    返回:
        labels: 聚类标签
        metrics: 评估指标字典
    """
    runner = MatlabRunner(method='auto')
    labels, metrics = runner.run_algorithm_pr(data, n_clusters, k)
    runner.close()
    return labels, metrics


if __name__ == "__main__":
    print("=" * 70)
    print("MATLAB集成测试")
    print("=" * 70)
    
    # 测试数据
    test_data = np.array([
        [1.0, 2.0, 0],
        [1.5, 2.5, 0],
        [5.0, 6.0, 1],
        [5.5, 6.5, 1]
    ])
    
    print("\n测试MATLAB调用...")
    runner = MatlabRunner(method='auto')
    print(f"使用方法: {runner.method}")
    
    if runner.method == 'subprocess':
        print("\n注意: 使用子进程方法需要:")
        print("1. 系统已安装MATLAB")
        print("2. MATLAB在系统PATH中")
        print("3. 创建MATLAB包装函数 (见下方说明)")
