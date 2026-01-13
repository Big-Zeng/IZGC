
% 临时脚本
try
    % 切换到工作目录
    cd('F:/科研/论文/分区论文/distrubuteArea/distributeAreaCode/algorithm_comparison');
    
    % 读取数据
    data = readmatrix('F:/科研/论文/分区论文/distrubuteArea/distributeAreaCode/algorithm_comparison/temp_matlab_data.csv');
    
    % 添加路径（使用绝对路径）
    addpath('F:/科研/论文/分区论文/distrubuteArea/distributeAreaCode/algorithm_comparison/compare/algorithm');
    
    % 运行算法
    [labels, ami, ari, fmi] = run_algorithm_pr_wrapper(data, 6, 9);
    
    % 保存结果
    save('F:/科研/论文/分区论文/distrubuteArea/distributeAreaCode/algorithm_comparison/temp_matlab_result.mat', 'labels', 'ami', 'ari', 'fmi');
    
    fprintf('MATLAB执行成功\n');
catch ME
    fprintf('MATLAB错误: %s\n', ME.message);
    % 保存错误信息
    error_msg = ME.message;
    save('F:/科研/论文/分区论文/distrubuteArea/distributeAreaCode/algorithm_comparison/temp_matlab_result.mat', 'error_msg');
end

% 退出
quit;
