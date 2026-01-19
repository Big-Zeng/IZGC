function [labels] = run_algorithm_pr_unsupervised(data, KK, k)
% MATLAB包装函数 - 无监督版本（无需真实标签）
% 
% 输入:
%   data - 数据矩阵 [x, y] (仅坐标，无标签)
%   KK - 簇数量
%   k - 近邻数量
%
% 输出:
%   labels - 聚类标签

% 提取数据（仅坐标）
B1 = data;

% 数据归一化
B = libsvmscale(B1, 0, 1);
[ND, dim] = size(B);

% 计算距离矩阵
data_features = B;
dist_row = pdist(data_features);
dist = squareform(dist_row);

% 初始化rho
rho = zeros(1, ND);

% 计算距离排序
[dist_sort, dist_order] = sort(dist, 2);

% 计算全局距离和方差
tk1 = sum(dist_sort(:, :), 2);
w = sum(tk1) / (0.5 * (ND - 1) * ND);
f1 = sqrt(sum(sum((dist - w).^2)) / (ND^2));

% 确保k不超过数据点数
k = min(k, ND - 1);

% 计算隶属模糊核
tempk1 = 1 ./ (dist_sort(:, 2:k) + 1).^2;
tempk2 = 1 ./ (dist_sort(:, k+1:end) * 1 + 1).^2;

tempke = exp(-(dist_sort(:, 2:k) .* dist_sort(:, 2:k) ./ 1));
tempkf = exp(-(dist_sort(:, k+1:end) .* dist_sort(:, k+1:end) * f1^2 ./ 1));

rho1 = sum(tempk1 .* tempke, 2)' / (k - 1);
rho2 = sum(tempk2 .* tempkf, 2)' / (ND - k + 1);
rho = rho1 + rho2;

% 计算相似度矩阵
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

% 计算共享近邻相似度
Pik_sorted = zeros(ND, ND);
Pik_order = zeros(ND, ND);
for i = 1:ND
    [Pik_sorted(:, i), Pik_order(:, i)] = sort(dist(:, i));
end

A2 = zeros(ND, ND);
for i = 1:ND-1
    for j = i+1:ND
        if Pik_sorted(k+1, i) + Pik_sorted(k+1, j) >= dist(i, j)
            c = intersect(Pik_order(2:k+1, i), Pik_order(2:k+1, j));
            A2(i, j) = size(c, 1);
        end
    end
end
A2 = (A2 + A2');
A2 = A2 / k;
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

% 计算gamma
gamma = rho .* delta;
[~, gamma_order] = sort(gamma, 'descend');

% 选择聚类中心
cccc = gamma_order(1:KK);

% 聚类分配
for i = 1:ND
    DD(i, i) = 0;
end

cl = ones(1, ND) * (-1);
for i = 1:KK
    cl(cccc(i)) = i;
end

DD(cccc, :) = zeros(size(cccc, 2), ND);

% 基于相似度分配
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

% 基于密度分配剩余点
if size(c, 2) < ND - 1
    for i = 1:ND
        if cl(ordrho(i)) == -1
            cl(ordrho(i)) = cl(nneigh(ordrho(i)));
        end
    end
end

% 输出标签
labels = cl';

end
