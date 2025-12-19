clc;
clear;
close all;
%% 读取数据
% AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\flame.txt');KK=2;
% AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Aggregation.txt');KK=7;
AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\pathbased.txt');KK=3;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\spiral.txt');KK=3;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\jain.txt');KK=2;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Compound.txt');KK=6;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\R15.txt');KK=15;
% AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\D31.txt');KK=31;
%%%%
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\s2-lab.txt');KK=15;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\birch1-lab.txt');KK=100;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\magic04.lab.txt');KK=14;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\Skin_NonSkin.txt');KK=14;
%AA=load('MLL1.mat');AA=AA.MLL1;KK=3;
%%%%%UCI数据集%%%%%%%%
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\iris.txt');KK=3;
% AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\seeds_dataset.txt');KK=3;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Wine.txt');KK=3;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Wdbc.txt');KK=2;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Segmentation.txt');KK=7;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Libras.txt');KK=15;
%AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Ionosphere.txt');KK=2;
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
%DPC密度峰聚类
%ret=FADPC(A,B,KK);
%ret=FADPC_H(B,KK);  
%ret=FADPC_DIV(B,KK);
%ret=DBSCAN(B,0.07,3);

 ret=DPC_t(B,KK);

%%%%%% 
F=[B,ret.cl'];
%if( dim==3)
ShowClusterA(F,'DPC Clustering');
%end
%icl=ret.icl;
%for i=1:length(icl)
 %   cl=icl(i);
 %   circle(B(cl,1),B(cl,2),ret.dc);
%    hold on;
%end

%画中心，二维
icl=ret.icl;
for i=1:length(icl)
   ccl=icl(i);
   scatter(B(ccl,1),B(ccl,2),100,'kh','MarkerFaceColor','w');%黑边六角星
   hold on;
end


LB=ret.cl';
critcurr1 = evalclusters(B,LB,'DaviesBouldin');

% [RI,ARI,NMI]=evolution(A,F,KK);
% disp(RI)
% disp(ARI)
% disp(NMI)

AMI(AA(:,end),ret.cl')
Library.GetAri(AA(:,end),ret.cl')
Library.GetFmi(AA(:,end),ret.cl')

