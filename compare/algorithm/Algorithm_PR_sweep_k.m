% Algorithm_PR k值扫描脚本
% 循环测试不同的k值，找到ARI最大的最优k值

clear;
close all;

% ==================== 配置区域 ====================
num = 94;  % 数据集编号
k_start = 50;  % k值起始
k_end = 100;  % k值结束
k_step = 1;  % k值步长
% ==================================================

% 加载数据集
switch num
    case{1},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\flame.txt');KK=2;
case{2},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\Aggregation.txt');KK=7;   
case{3},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\pathbased.txt');KK=3;
case{4},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\spiral.txt');KK=3;
case{5},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\jain.txt');KK=2;
case{6},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\Compound.txt');KK=6;%
case{7},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\R15.txt');KK=15;
case{8},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\D31.txt');KK=31;

case{91},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\sizes53.mat');KK=4;AA=AA.AA;    
case{92},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\cheng_data\artificial\five_cluster.mat');KK=5;AA=AA.d;        
case{93},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\sn3.mat');KK=3;AA=AA.d; 
case{94},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\creat\ring.txt');KK=2;    
case{95},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\creat\Zigzag.txt');KK=3;
%case{100},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\data\OlivettiFace.mat');KK=40;AA=AA.d;         
        
%%%%
case{50},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\circle3.mat'); KK=3;AA = AA.d;      
case{51},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\cth3.mat'); KK=4;AA = AA.d;
case{52},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\db13.mat'); KK=4;AA = AA.d;    
case{53},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\jain23.mat'); KK=2;AA = AA.d;     
case{54},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\line4.mat'); KK=4;AA = AA.d; %dpc达到1     
case{55},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\newDatasets\ls3.mat'); KK=6;AA = AA.d;     
    
case{56},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\10-19dataset\shape\Eyes2.mat'); KK=3;AA = AA.d;       
    case{57},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\lineblobs.txt'); KK=3; 
case{58},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\cheng_data\artificial\Twomoons.mat'); KK=2;AA = AA.d;      
case{59},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\10-19dataset\shape\Sticks2.mat'); KK=4;AA = AA.d;      
case{60},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\10-19dataset\shape\threecircles2.mat'); KK=3;AA = AA.d;
case{61},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\10-19dataset\shape\spiral2.mat'); KK=2;AA = AA.d;    
case{62},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\cheng_data\artificial\spiral_unbalance.mat'); KK=2;AA = AA.d;
    
case{61},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\s1.mat');AA=AA.d ;KK=15;
case{62},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\large2d\s2-lab.txt');KK=15;
case{63},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\s3.mat');AA=AA.d ;KK=15;    
case{64},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\shape\s4.mat');AA=AA.d ;KK=15;    

    
case{10},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\large2d\birch1-lab.txt');KK=100; % pdist出问题，距离数组太大
case{11},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\large2d\magic04.lab.txt');KK=14;
case{12},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\large2d\Skin_NonSkin.txt');KK=14; % pdist出问题，距离数组太大

case{13},AA=load('MLL1.mat');AA=AA.MLL1;KK=3;
%%%%%UCI数据集%%%%%%%%
case{14},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\iris.txt');KK=3;
case{15},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\seeds_dataset.txt');KK=3;
case{16},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Wine.txt');KK=3;
case{17},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Wdbc.txt');KK=2;
case{18},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Segmentation.txt');KK=7;
case{19},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Libras.txt');KK=15;
case{20},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Ionosphere.txt');KK=2;
case{23},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\WaveformDatabaseGenerator\waveform.mat');AA=AA.DD;KK=3;     
case{24},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Ecoli\ecoli.data');KK=8;
case{25},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\uci\Dermatology\Dermatology.mat');AA=AA.d;KK=6;

case{30},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\parkinsons1.txt');KK=2;
case{31},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\balance-scale.txt');KK=3;
case{32},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\secom.txt');KK=2;
case{33},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\Internet Advertisements.txt');KK=max(AA(end,:));
case{34},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\Phishing Websites.txt');KK=max(AA(end,:));
case{35},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\Online Shoppers Purchasing Intention .txt');KK=max(AA(end,:)); 
case{36},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\pendigits.txt');KK=10;
case{37},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\dataset\NEWuci\RingNorm.txt');KK=2;    
end

fprintf('========================================\n');
fprintf('Algorithm_PR k值扫描测试\n');
fprintf('========================================\n');
fprintf('数据集编号: %d\n', num);
fprintf('真实簇数: %d\n', KK);
fprintf('k值范围: %d - %d (步长: %d)\n', k_start, k_end, k_step);
fprintf('========================================\n\n');

% 数据预处理
B1 = AA(:, 1:end-1); 
B = libsvmscale(B1, 0, 1);
[rows, dim] = size(AA);
A = [B, AA(:, end)];
LAB = AA(:, end);
ND = size(B, 1);

% 计算距离矩阵（只需计算一次）
data = B;
dist_row = pdist(data);
dist = squareform(dist_row);

% 存储结果
k_values = k_start:k_step:k_end;
n_tests = length(k_values);
results = zeros(n_tests, 4);  % [k, AMI, ARI, FMI]

% 循环测试不同的k值
fprintf('开始扫描k值...\n');
fprintf('%-6s %-10s %-10s %-10s %-10s\n', '序号', 'k值', 'AMI', 'ARI', 'FMI');
fprintf('--------------------------------------------------------\n');

for idx = 1:n_tests
    k = k_values(idx);
    K = k;
    
    try
        % ========== 核心算法部分 ==========
        % 初始化rho
        rho = zeros(1, ND);
        
        % 密度模型计算
        [dist_sort, dist_order] = sort(dist, 2);
        
        tk1 = sum(dist_sort(:, :), 2);
        w = sum(tk1) / (0.5 * (ND - 1) * ND);
        f1 = sqrt(sum(sum((dist - w).^2)) / (ND^2));
        
        tempk1 = 1 ./ (dist_sort(:, 2:k) + 1).^2;
        tempk2 = 1 ./ (dist_sort(:, k+1:end) * 1 + 1).^2;
        
        tempke = exp(-(dist_sort(:, 2:k) .* dist_sort(:, 2:k) ./ 1));
        tempkf = exp(-(dist_sort(:, k+1:end) .* dist_sort(:, k+1:end) * f1^2 ./ 1));
        
        rho1 = sum(tempk1 .* tempke, 2)' / (k - 1);
        rho2 = sum(tempk2 .* tempkf, 2)' / (ND - k + 1);
        rho = rho1 + rho2;
        
        % 计算u矩阵
        temk1 = 1 ./ (dist_sort(:, :) + 1).^2;
        temk2 = 1 ./ (dist_sort(:, :) + 1).^2;
        
        temke = exp(-(dist_sort(:, :) .* dist_sort(:, :)));
        temkf = exp(-(dist_sort(:, :) .* dist_sort(:, :) * f1^2));
        
        u_knn = temk1 .* temke;
        u_not = temk2 .* temkf;
        
        u_knn(:, [1, k+1:end]) = 0;
        u_not(:, 1:k) = 0;
        uu = u_knn + u_not;
        
        u = zeros(ND, ND);
        for i = 1:ND
            u(i, dist_order(i, :)) = uu(i, :);
        end
        
        % 共享近邻相似度
        Pik_sorted = zeros(ND, ND);
        Pik_order = zeros(ND, ND);
        for i = 1:ND
            [Pik_sorted(:, i), Pik_order(:, i)] = sort(dist(:, i));
        end
        
        A2 = zeros(ND, ND);
        for i = 1:ND-1
            for j = i+1:ND
                if Pik_sorted(K+1, i) + Pik_sorted(K+1, j) >= dist(i, j)
                    c = intersect(Pik_order(2:K+1, i), Pik_order(2:K+1, j));
                    A2(i, j) = size(c, 1);
                end
            end
        end
        A2 = (A2 + A2');
        A2 = A2 / K;
        DDt = A2;
        
        DD = DDt .* u;
        DD(isnan(DD)) = 0;
        
        % 计算delta
        maxd = max(max(dist));
        [~, ordrho] = sort(rho, 'descend');
        delta = zeros(1, ND);
        nneigh = zeros(1, ND);
        delta(ordrho(1)) = -1;
        nneigh(ordrho(1)) = 0;
        
        for ii = 2:ND
            delta(ordrho(ii)) = maxd;
            for jj = 1:ii-1
                if dist(ordrho(ii), ordrho(jj)) < delta(ordrho(ii))
                    delta(ordrho(ii)) = dist(ordrho(ii), ordrho(jj));
                    nneigh(ordrho(ii)) = ordrho(jj);
                end
            end
        end
        delta(ordrho(1)) = max(delta(:));
        
        % 归一化
        rho = mapminmax(rho, 0, 1);
        delta = mapminmax(delta, 0, 1);
        
        % 计算gamma并选择聚类中心
        gamma = rho .* delta;
        [~, gamma_order] = sort(gamma, 'descend');
        class_num = KK;
        cccc = gamma_order(1:class_num);
        
        % 基于密度聚类
        for i = 1:ND
            DD(i, i) = 0;
        end
        cl = ones(1, ND) * (-1);
        for i = 1:KK
            cl(cccc(i)) = i;
        end
        DD(cccc, :) = zeros(size(cccc, 2), ND);
        
        while 1
            c = find(cl > 0);
            c1 = max(max(DD(:, c)));
            if c1 == 0
                break
            end
            while 1
                [a, b] = find(DD(:, c) == c1);
                if cl(a(1)) < 0
                    cl(a(1)) = cl(c(b(1)));
                    DD(a(1), :) = zeros(1, ND);
                    break
                else
                    DD(a(1), c(b(1))) = 0;
                end
            end
            if min(cl) == 1
                break
            end
            if max(max(DD)) == 0
                break
            end
        end
        
        % 分配剩余点
        if size(c, 2) < ND - 1
            for i = 1:ND
                if cl(ordrho(i)) == -1
                    cl(ordrho(i)) = cl(nneigh(ordrho(i)));
                end
            end
        end
        
        % 计算评价指标
        AMI_val = AMI(AA(:, end), cl');
        ARI_val = Library.GetAri(AA(:, end), cl');
        FMI_val = Library.GetFmi(AA(:, end), cl');
        
        % 存储结果
        results(idx, :) = [k, AMI_val, ARI_val, FMI_val];
        
        fprintf('%-6d %-10d %-10.4f %-10.4f %-10.4f\n', idx, k, AMI_val, ARI_val, FMI_val);
        
    catch ME
        fprintf('%-6d %-10d 失败: %s\n', idx, k, ME.message);
        results(idx, :) = [k, 0, 0, 0];
    end
end

fprintf('========================================\n\n');

% 找到最优k值
[max_ARI, max_idx] = max(results(:, 3));
best_k = results(max_idx, 1);
best_AMI = results(max_idx, 2);
best_FMI = results(max_idx, 4);

fprintf('最优结果（ARI最大）:\n');
fprintf('========================================\n');
fprintf('最优k值: %d\n', best_k);
fprintf('AMI:     %.4f\n', best_AMI);
fprintf('ARI:     %.4f\n', max_ARI);
fprintf('FMI:     %.4f\n', best_FMI);
fprintf('========================================\n\n');

% 显示前5名
fprintf('前5名结果:\n');
[~, sorted_idx] = sort(results(:, 3), 'descend');
fprintf('%-6s %-10s %-10s %-10s %-10s\n', '排名', 'k值', 'AMI', 'ARI', 'FMI');
fprintf('--------------------------------------------------------\n');
for i = 1:min(5, n_tests)
    idx = sorted_idx(i);
    fprintf('%-6d %-10d %-10.4f %-10.4f %-10.4f\n', ...
        i, results(idx, 1), results(idx, 2), results(idx, 3), results(idx, 4));
end
fprintf('\n');

% 绘制结果图
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(results(:, 1), results(:, 2), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('k值');
ylabel('AMI');
title('AMI vs k值');
grid on;
hold on;
plot(best_k, best_AMI, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
legend('AMI', sprintf('最大值 (k=%d)', best_k));

subplot(1, 3, 2);
plot(results(:, 1), results(:, 3), 'g-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('k值');
ylabel('ARI');
title('ARI vs k值');
grid on;
hold on;
plot(best_k, max_ARI, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
legend('ARI', sprintf('最大值 (k=%d)', best_k));

subplot(1, 3, 3);
plot(results(:, 1), results(:, 4), 'm-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('k值');
ylabel('FMI');
title('FMI vs k值');
grid on;
hold on;
plot(best_k, best_FMI, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
legend('FMI', sprintf('最大值 (k=%d)', best_k));

sgtitle(sprintf('Algorithm_PR k值扫描结果 (数据集 %d)', num));

% 保存结果到文件
filename = sprintf('Algorithm_PR_k_sweep_dataset%d.txt', num);
fid = fopen(filename, 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'Algorithm_PR k值扫描测试结果\n');
fprintf(fid, '========================================\n');
fprintf(fid, '数据集编号: %d\n', num);
fprintf(fid, '真实簇数: %d\n', KK);
fprintf(fid, 'k值范围: %d - %d (步长: %d)\n', k_start, k_end, k_step);
fprintf(fid, '测试数量: %d\n\n', n_tests);

fprintf(fid, '详细结果:\n');
fprintf(fid, '%-10s %-10s %-10s %-10s\n', 'k值', 'AMI', 'ARI', 'FMI');
fprintf(fid, '--------------------------------------------\n');
for i = 1:n_tests
    fprintf(fid, '%-10d %-10.4f %-10.4f %-10.4f\n', ...
        results(i, 1), results(i, 2), results(i, 3), results(i, 4));
end

fprintf(fid, '\n========================================\n');
fprintf(fid, '最优结果（ARI最大）:\n');
fprintf(fid, '========================================\n');
fprintf(fid, '最优k值: %d\n', best_k);
fprintf(fid, 'AMI:     %.4f\n', best_AMI);
fprintf(fid, 'ARI:     %.4f\n', max_ARI);
fprintf(fid, 'FMI:     %.4f\n', best_FMI);
fprintf(fid, '========================================\n');
fclose(fid);

fprintf('结果已保存到: %s\n', filename);
