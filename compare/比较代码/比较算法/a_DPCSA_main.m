% clc;
clear;
close all;
% 20-9-23 调用DPCSAF 实现DPCSA；不需要参数；指标和y论文一致

num = 3   ;

switch  num
case{1},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\flame.txt');KK=2;
case{2},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Aggregation.txt');KK=7;   
case{3},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\pathbased.txt');KK=3;
case{4},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\spiral.txt');KK=3;
case{5},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\jain.txt');KK=2;
case{6},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Compound.txt');KK=6;%
case{7},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\R15.txt');KK=15;
case{8},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\D31.txt');KK=31;

%%%%%UCI数据集%%%%%%%%
case{14},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\iris.txt');KK=3;
case{15},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\seeds_dataset.txt');KK=3;
case{16},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Wine.txt');KK=3;
case{17},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Wdbc.txt');KK=2;
case{18},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Segmentation.txt');KK=7;
case{19},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Libras.txt');KK=15;
case{20},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Ionosphere.txt');KK=2;
case{23},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\WaveformDatabaseGenerator\waveform.mat');AA=AA.DD;KK=3;     
case{24},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Ecoli\ecoli.data');KK=8;
case{25},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Dermatology\Dermatology.mat');AA=AA.d;KK=6;
   
    
end
%% 数据列归一化 
B=AA(:,1:end-1);
B= libsvmscale(B,0,1);  % 数据归一化
[rows,dim]=size(AA);
A=[B,AA(:,end)];
LAB=AA(:,end);
%% 显示原始结果 
%数据格式为三列，前两列是二维数据，最后一列是类别  x,y,c  
%最多显示7中类别的聚类
%  if( dim==3)  %改维数
    ShowClusterA(A,'origin graphic')
%  end

%% 比较
%%% 算所有点的距离
data = B;                   %除去最后一行标签，归一化后的数据
dist_row = pdist(data);     %pdist（data）计算 data 中 各对行向量 的相互距离 存储为一个行向量
dist=squareform(dist_row);  % squareform转成对 角线为零的 距离矩阵

ret=DPCSAF(dist,KK);


%%%%%% 

F=[B,ret.cl'];
%if( dim==3)
ShowClusterA(F,' ');

%画中心
for i=1:length(ret.icl)
   ccl=ret.icl(i);
%    scatter(B(ccl,1),B(ccl,2),'p','k');%黑边六角星
   scatter(B(ccl,1),B(ccl,2),100,'kp','MarkerFaceColor','w');%黑边五角星   
   hold on;
end

% LB=cl';
% critcurr1 = evalclusters(B,LB,'DaviesBouldin');
% disp(critcurr1) ;
% [RI,ARI,NMI]=evolution(A,F,KK);
cl = ret.cl ;
AMI = GetAmi(AA(:,end),cl');
ARI = Library.GetAri(AA(:,end),cl');
FMI = Library.GetFmi(AA(:,end),cl');
% disp(RI)
% disp(NMI)
ev = [AMI ,ARI,FMI] ;
disp(ev)


