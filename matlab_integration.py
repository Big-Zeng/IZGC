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
        # Try MATLAB Engine
        try:
            import matlab.engine
            print("检测到 MATLAB Engine API")
            return 'engine'
        except ImportError:
            pass
        
        # Try Oct2Py
        try:
            import oct2py
            print("检测到 Oct2Py (Octave)")
            return 'octave'
        except ImportError:
            pass
        
        # Fall back to a subprocess
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
        
        # Convert the data to MATLAB format
        matlab_data = matlab.double(data.tolist())
        
        # Add the MATLAB path
        self.matlab_engine.addpath('compare/algorithm', nargout=0)
        
        # Call the MATLAB function
        try:
            # A simplified MATLAB wrapper is needed to call Algorithm_PR
            labels, ami, ari, fmi = self.matlab_engine.run_algorithm_pr_wrapper(
                matlab_data, float(n_clusters), float(k), nargout=4
            )
            
            # Convert back to Python format
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
        # Add the path
        self.octave_engine.addpath('compare/algorithm')
        
        try:
            # Call the MATLAB function
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
    
    def run_algorithm_pr_unsupervised(self, data, n_clusters, k):
        """
        运行MATLAB的Algorithm_PR算法（无监督版本）
        
        参数:
            data: numpy数组，shape=(n_samples, 2)，列为[x, y]（仅坐标，无标签）
            n_clusters: 簇数量
            k: 近邻数量
            
        返回:
            labels: 聚类标签
        """
        if self.method == 'engine':
            return self._run_unsupervised_with_engine(data, n_clusters, k)
        elif self.method == 'octave':
            return self._run_unsupervised_with_octave(data, n_clusters, k)
        else:
            return self._run_unsupervised_with_subprocess(data, n_clusters, k)
    
    def _run_unsupervised_with_engine(self, data, n_clusters, k):
        """使用MATLAB Engine运行（无监督版本）"""
        import matlab
        
        # Convert the data to MATLAB format
        matlab_data = matlab.double(data.tolist())
        
        # Add the MATLAB path
        self.matlab_engine.addpath('compare/algorithm', nargout=0)
        
        try:
            # Call the unsupervised MATLAB variant
            labels = self.matlab_engine.run_algorithm_pr_unsupervised(
                matlab_data, float(n_clusters), float(k), nargout=1
            )
            
            # Convert back to Python format
            labels = np.array(labels).flatten()
            return labels
        except Exception as e:
            print(f"MATLAB执行错误: {e}")
            return None
    
    def _run_unsupervised_with_octave(self, data, n_clusters, k):
        """使用Octave运行（无监督版本）"""
        # Add the path
        self.octave_engine.addpath('compare/algorithm')
        
        try:
            # Call the unsupervised MATLAB variant
            labels = self.octave_engine.run_algorithm_pr_unsupervised(
                data, n_clusters, k, nout=1
            )
            return labels.flatten()
        except Exception as e:
            print(f"Octave执行错误: {e}")
            return None
    
    def _run_unsupervised_with_subprocess(self, data, n_clusters, k):
        """使用子进程运行MATLAB（无监督版本）"""
        # Directory holding matlab_integration.py (the project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        algorithm_dir = os.path.join(script_dir, 'compare', 'algorithm')
        
        # Ensure the directory exists
        if not os.path.exists(algorithm_dir):
            print(f"错误: 目录不存在 {algorithm_dir}")
            return None
        
        # Keep temporary files under algorithm/ to avoid non-ASCII paths
        temp_data_file = os.path.join(algorithm_dir, 'temp_data_unsup.csv')
        result_file = os.path.join(algorithm_dir, 'temp_result_unsup.mat')
        temp_script_file = os.path.join(algorithm_dir, 'temp_script_unsup.m')
        
        # Save the data (coordinates only, no labels)
        np.savetxt(temp_data_file, data, delimiter=',')
        
        # Create a temporary MATLAB script
        matlab_script = f"""
% 临时脚本 - 无监督版本
try
    % 读取数据（仅坐标）
    data = readmatrix('temp_data_unsup.csv');
    
    % 运行无监督算法
    labels = run_algorithm_pr_unsupervised(data, {n_clusters}, {k});
    
    % 保存结果
    save('temp_result_unsup.mat', 'labels');
    
    fprintf('MATLAB执行成功\\n');
catch ME
    fprintf('MATLAB错误: %s\\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s, 行 %d\\n', ME.stack(1).name, ME.stack(1).line);
    end
    % 保存错误信息
    error_msg = ME.message;
    save('temp_result_unsup.mat', 'error_msg');
end

quit;
"""
        
        with open(temp_script_file, 'w', encoding='utf-8') as f:
            f.write(matlab_script)
        
        try:
            # Run MATLAB from the algorithm directory
            result = subprocess.run(
                ['matlab', '-batch', "run('temp_script_unsup.m')"], 
                check=True, 
                capture_output=True,
                text=True,
                cwd=algorithm_dir
            )
            
            # Check whether the result file exists
            if not os.path.exists(result_file):
                print(f"MATLAB未生成结果文件")
                print(f"MATLAB stdout: {result.stdout}")
                print(f"MATLAB stderr: {result.stderr}")
                return None
            
            # Read the results
            from scipy.io import loadmat
            mat_result = loadmat(result_file)
            
            # Check for errors
            if 'error_msg' in mat_result:
                print(f"MATLAB内部错误: {mat_result['error_msg']}")
                return None
            
            labels = mat_result['labels'].flatten()
            return labels
            
        except subprocess.CalledProcessError as e:
            print(f"子进程执行错误: {e}")
            print(f"MATLAB stdout: {e.stdout}")
            print(f"MATLAB stderr: {e.stderr}")
        except Exception as e:
            print(f"执行错误: {e}")
        finally:
            # Clean up temporary files
            for f in [temp_data_file, temp_script_file, result_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
        
        return None

    def _run_with_subprocess(self, data, n_clusters, k):
        """使用子进程运行MATLAB"""
        # Directory holding matlab_integration.py (the project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        algorithm_dir = os.path.join(script_dir, 'compare', 'algorithm')
        
        # Ensure the directory exists
        if not os.path.exists(algorithm_dir):
            print(f"错误: 目录不存在 {algorithm_dir}")
            return None, None
        
        # Keep temporary files under algorithm/ to avoid non-ASCII paths
        temp_data_file = os.path.join(algorithm_dir, 'temp_data.csv')
        result_file = os.path.join(algorithm_dir, 'temp_result.mat')
        temp_script_file = os.path.join(algorithm_dir, 'temp_script.m')
        
        # Save the data
        np.savetxt(temp_data_file, data, delimiter=',')
        
        # Create a temporary MATLAB script under algorithm/
        matlab_script = f"""
% 临时脚本 - 在algorithm目录下运行
try
    % 读取数据
    data = readmatrix('temp_data.csv');
    
    % 运行算法
    [labels, ami, ari, fmi] = run_algorithm_pr_wrapper(data, {n_clusters}, {k});
    
    % 保存结果
    save('temp_result.mat', 'labels', 'ami', 'ari', 'fmi');
    
    fprintf('MATLAB执行成功\\n');
catch ME
    fprintf('MATLAB错误: %s\\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s, 行 %d\\n', ME.stack(1).name, ME.stack(1).line);
    end
    % 保存错误信息
    error_msg = ME.message;
    save('temp_result.mat', 'error_msg');
end

quit;
"""
        
        with open(temp_script_file, 'w', encoding='utf-8') as f:
            f.write(matlab_script)
        
        try:
            # Run MATLAB from the algorithm directory
            result = subprocess.run(
                ['matlab', '-batch', "run('temp_script.m')"], 
                check=True, 
                capture_output=True,
                text=True,
                cwd=algorithm_dir
            )
            
            # Check whether the result file exists
            if not os.path.exists(result_file):
                print(f"MATLAB未生成结果文件")
                print(f"MATLAB stdout: {result.stdout}")
                print(f"MATLAB stderr: {result.stderr}")
                return None, None
            
            # Read the results
            from scipy.io import loadmat
            mat_result = loadmat(result_file)
            
            # Check for errors
            if 'error_msg' in mat_result:
                print(f"MATLAB内部错误: {mat_result['error_msg']}")
                return None, None
            
            labels = mat_result['labels'].flatten()
            ami = float(mat_result['ami'][0, 0])
            ari = float(mat_result['ari'][0, 0])
            fmi = float(mat_result['fmi'][0, 0])
            
            return labels, {'AMI': ami, 'ARI': ari, 'FMI': fmi}
            
        except subprocess.CalledProcessError as e:
            print(f"子进程执行错误: {e}")
            print(f"MATLAB stdout: {e.stdout}")
            print(f"MATLAB stderr: {e.stderr}")
        except Exception as e:
            print(f"执行错误: {e}")
        finally:
            # Clean up temporary files
            for f in [temp_data_file, temp_script_file, result_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
        
        return None, None
    
    def close(self):
        """关闭引擎"""
        if self.matlab_engine:
            self.matlab_engine.quit()
        if self.octave_engine:
            self.octave_engine.exit()


# Simplified calling interface
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


def run_matlab_algorithm_pr_unsupervised(data, n_clusters, k=11):
    """
    简化的接口，运行MATLAB的Algorithm_PR算法（无监督版本）
    
    参数:
        data: numpy数组，shape=(n_samples, 2)，列为[x, y]（仅坐标）
        n_clusters: 簇数量
        k: 近邻数量
        
    返回:
        labels: 聚类标签
    """
    runner = MatlabRunner(method='auto')
    labels = runner.run_algorithm_pr_unsupervised(data, n_clusters, k)
    runner.close()
    return labels


if __name__ == "__main__":
    print("=" * 70)
    print("MATLAB集成测试")
    print("=" * 70)
    
    # Test data
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
