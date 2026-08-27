
% 临时脚本 - 无监督版本
try
    % 读取数据（仅坐标）
    data = readmatrix('temp_data_unsup.csv');
    
    % 运行无监督算法
    labels = run_algorithm_pr_unsupervised(data, 3, 4);
    
    % 保存结果
    save('temp_result_unsup.mat', 'labels');
    
    fprintf('MATLAB执行成功\n');
catch ME
    fprintf('MATLAB错误: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s, 行 %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    % 保存错误信息
    error_msg = ME.message;
    save('temp_result_unsup.mat', 'error_msg');
end

quit;
