clc;
clear;
close all;
%% 读取数据
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\flame.txt');KK=2;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\Aggregation.txt');KK=7;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\pathbased.txt');KK=3;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\spiral.txt');KK=3;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\jain.txt');KK=2;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\Compound.txt');KK=6;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\R15.txt');KK=15;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\shape\D31.txt');KK=31;
%%%%
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\large2d\s2-lab.txt');KK=15;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\large2d\birch1-lab.txt');KK=100;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\large2d\magic04.lab.txt');KK=14;
%AA=load('C:\Users\wangy\Desktop\cluster-master\dataset\large2d\Skin_NonSkin.txt');K=14;
%AA=load('MLL1.mat');AA=AA.MLL1;KK=3;
%%%%%UCI数据集%%%%%%%%
%AA=load('C:\Users\wangy\Desktop\DPC\dataset\uci\iris.txt');KK=3;
%AA=load('C:\Users\wangy\Desktop\DPC\dataset\uci\seeds_dataset.txt');KK=3;
AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\Wine.txt');KK=3;
%AA=load('C:\Users\wangy\Desktop\DPC\dataset\uci\Wdbc.txt');KK=2;
%AA=load('C:\Users\wangy\Desktop\DPC\dataset\uci\Segmentation.txt');KK=7;
%AA=load('C:\Users\wangy\Desktop\DPC\dataset\uci\Libras.txt');KK=15;
%AA=load('C:\Users\wangy\Desktop\DPC\dataset\uci\Ionosphere.txt');KK=2;
%% 数据列归一化 
B=AA(:,1:end-1);
B= libsvmscale(B,0,1);  % 数据归一化
[n,~]=size(B);


dc=0.2212;
for i=1:n
   fa(i)=0;
   for j=1:n
       if i == j
           fa(i)=fa(i);
       else
      fa(i)=fa(i)+exp(-(norm(B(i,:)-B(j,:))/dc)^2);
       end
   end
end
fa=sort(fa);
x=1:n;
figure(1);
plot(x,fa,'.');
title('wine')