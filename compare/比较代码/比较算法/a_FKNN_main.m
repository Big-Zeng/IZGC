close all;clear;%clc;
%% 读取数据

num = 3       ;

K =  9         ;%5
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
a=find(AA(:,end)==0);
AA(a,end)=3;
data=AA(:,1:end-1);
data= libsvmscale(data,0,1);  % 数据归一化
[rows,dim]=size(AA);
A=[data,AA(:,end)];

   ShowClusterA(A,'origin graphic')
answer=AA(:,end);
dist1=pdist2(data,data,'minkowski',2);
N=size(data,1);
%dist1=pdist(data);

%Execution
% K=2:50;
% 
% for p=1:length(K)
%   %If you want to choose centers manually, set AutoPick to 0, otherwise, number of centers
%   tic
%   result(p)=KnnDpc(data,answer,6,'AutoPick',KK,'Distance',dist1,'Ui',true);
%   toc
%   t(p)=toc;
% %   fprintf('K=%i,AMI=%f,ARI=%f,FMI=%f\n',K(p),result(p).ami,result(p).ari,result(p).fmi);
% end

for p=1:length(K)
  %If you want to choose centers manually, set AutoPick to 0, otherwise, number of centers
  for i=1
  tic
  result(p)=KnnDpc(data,answer,K(p),'AutoPick',KK,'Distance',dist1,'Ui',true);
  toc
  t(i)=toc;
  end
%   fprintf('K=%i,AMI=%f,ARI=%f,FMI=%f\n',K(p),result(p).ami,result(p).ari,result(p).fmi);
end
F=[data,result.cluster];
% [RI,ARI,NMI]=evolution(A,F,KK);
% acc=ClusteringAccuracy(result.cluster',AA(:,end));
% acc
% RI
% B=AA(:,1:2);
% F=[data,result.cluster];
% if( dim==3)
% ShowClusterA(F,'DPC-CNN-MCM on Pathbased');
% hold on
% scatter(data(result.center,1),data(result.center,2),'h','k');
% end

%% 画图
% F=[B,cl'];
%if( dim==3)
ShowClusterA(F,' ');   

%画中心
% for i=1:length(icl)
%    ccl=icl(i);
   scatter(data(result.center,1),data(result.center,2),100,'kp','MarkerFaceColor','w');%黑边六角星
%       scatter(B(ccl,1),B(ccl,2),100,'kp','MarkerFaceColor','w');%黑边五角星
%    scatter(B(ccl,1),B(ccl,2),100,'p','k');%黑边五角星
%    hold on;
% end

dashboard=table([result(:).ami]',[result(:).ari]',[result(:).fmi]',K','VariableNames',{'AMI','ARI','FMI','K'});
resultBest=dashboard(dashboard.ARI==max([result(:).ari]),:)

