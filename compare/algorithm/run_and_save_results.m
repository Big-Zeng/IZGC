% run_and_save_results.m
% 批量运行数据集并保存结果
% 用于在MATLAB中运行Algorithm_PR，然后在Python中读取结果

clear;
close all;

% 创建结果目录
if ~exist('../matlab_results', 'dir')
    mkdir('../matlab_results');
end

% 定义数据集列表
datasets = {
    'flame', 2;
    'Aggregation', 7;
    'jain', 2;
    'spiral', 3;
    'pathbased', 3;
    'Compound', 6;
    'R15', 15
};

% 存储所有结果
all_results = struct();

fprintf('========================================\n');
fprintf('批量运行MATLAB Algorithm_PR\n');
fprintf('========================================\n\n');

% 遍历每个数据集
for i = 1:size(datasets, 1)
    dataset_name = datasets{i, 1};
    KK = datasets{i, 2};  % 簇数
    
    fprintf('处理数据集 %d/%d: %s (簇数=%d)\n', i, size(datasets, 1), dataset_name, KK);
    
    try
        % 读取数据
        data_file = sprintf('../../data/%s.csv', dataset_name);
        
        % 检查文件是否存在
        if ~exist(data_file, 'file')
            fprintf('  警告: 文件不存在 %s\n', data_file);
            continue;
        end
        
        % 读取CSV文件
        data_table = readtable(data_file);
        
        % 转换为矩阵 [x, y, label]
        AA = [data_table.x, data_table.y, data_table.target];
        
        % 运行Algorithm_PR
        k = 11;  % 近邻数
        fprintf('  运行Algorithm_PR (k=%d)...\n', k);
        
        [labels, ami, ari, fmi] = run_algorithm_pr_wrapper(AA, KK, k);
        
        % 保存结果到结构体
        all_results.(dataset_name).labels = labels;
        all_results.(dataset_name).AMI = ami;
        all_results.(dataset_name).ARI = ari;
        all_results.(dataset_name).FMI = fmi;
        all_results.(dataset_name).n_clusters = KK;
        all_results.(dataset_name).n_samples = size(AA, 1);
        
        % 保存标签到CSV文件
        labels_file = sprintf('../matlab_results/%s_labels.csv', dataset_name);
        csvwrite(labels_file, labels);
        
        % 打印结果
        fprintf('  结果: AMI=%.4f, ARI=%.4f, FMI=%.4f\n', ami, ari, fmi);
        fprintf('  标签已保存到: %s\n\n', labels_file);
        
    catch ME
        fprintf('  错误: %s\n', ME.message);
        fprintf('  跳过此数据集\n\n');
        continue;
    end
end

% 保存所有结果到MAT文件
mat_file = '../matlab_results/all_results.mat';
save(mat_file, 'all_results');
fprintf('所有结果已保存到: %s\n', mat_file);

% 导出为JSON格式（便于Python读取）
json_file = '../matlab_results/results.json';
try
    % MATLAB R2016b及以后版本支持jsonencode
    jsonStr = jsonencode(all_results);
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    fprintf('JSON结果已保存到: %s\n', json_file);
catch
    fprintf('警告: 无法保存JSON文件（需要MATLAB R2016b或更高版本）\n');
end

% 生成汇总报告
fprintf('\n========================================\n');
fprintf('汇总报告\n');
fprintf('========================================\n');
fprintf('%-15s %8s %8s %8s %8s\n', '数据集', 'AMI', 'ARI', 'FMI', '样本数');
fprintf('----------------------------------------\n');

dataset_names = fieldnames(all_results);
for i = 1:length(dataset_names)
    name = dataset_names{i};
    result = all_results.(name);
    fprintf('%-15s %8.4f %8.4f %8.4f %8d\n', ...
        name, result.AMI, result.ARI, result.FMI, result.n_samples);
end

fprintf('========================================\n');
fprintf('完成！\n');
fprintf('现在可以在Python中使用这些结果了。\n');
fprintf('运行: python compareTest_with_saved_matlab.py\n');
fprintf('========================================\n');
