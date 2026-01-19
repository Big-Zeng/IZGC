
% 临时脚本 - 在algorithm目录下运行
try
    % 读取数据
    data = readmatrix('temp_data.csv');
    
    % 运行算法
    [labels, ami, ari, fmi] = run_algorithm_pr_wrapper(data, 10, 8);
    
    % 保存结果
    save('temp_result.mat', 'labels', 'ami', 'ari', 'fmi');
    
    fprintf('MATLAB执行成功\n');
catch ME
    fprintf('MATLAB错误: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s, 行 %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    % 保存错误信息
    error_msg = ME.message;
    save('temp_result.mat', 'error_msg');
end

quit;
