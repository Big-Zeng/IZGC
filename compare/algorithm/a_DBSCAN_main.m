clc;
clear;
close all;
% 0 is reserved for NOISE
num = 4    ;

R =       0.07       ;     %0.05
MinPts = 5        ;        %4     

%ret=DBSCAN(B,0.04,6);
switch  num
    
case{1},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\flame.txt');KK=2;
case{2},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Aggregation.txt');KK=7;   
case{3},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\pathbased.txt');KK=3;
case{4},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\spiral.txt');KK=3;
case{5},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\jain.txt');KK=2;
case{6},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Compound.txt');KK=6;%
case{7},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\R15.txt');KK=15;
case{8},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\D31.txt');KK=31;

case{96},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\t4.8k.txt');KK=6  ;    
case{110},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\DLORE-DP-master\synthetic data sets\d62.txt');KK=4 ;
    
case{50},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\circle3.mat'); KK=3;AA = AA.d;      
case{51},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\cth3.mat'); KK=4;AA = AA.d;
case{52},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\db13.mat'); KK=4;AA = AA.d;    
case{53},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\jain23.mat'); KK=2;AA = AA.d;     
case{54},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\line4.mat'); KK=4;AA = AA.d;      
case{55},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\ls3.mat'); KK=6;AA = AA.d;     
    
case{56},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\shape\Eyes2.mat'); KK=3;AA = AA.d;       
case{57},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\shape\LineBlobs2.mat'); KK=3;AA = AA.d; 
case{58},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\cheng_data\artificial\Twomoons.mat'); KK=2;AA = AA.d;      
case{59},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\shape\Sticks2.mat'); KK=4;AA = AA.d;      
case{60},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\shape\threecircles2.mat'); KK=3;AA = AA.d;
case{61},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\shape\spiral2.mat'); KK=2;AA = AA.d;    
case{62},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\cheng_data\artificial\spiral_unbalance.mat'); KK=2;AA = AA.d;      
           
%%%%
case{81},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\s1.mat');AA=AA.d ;KK=15;
case{82},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\s2-lab.txt');KK=15;
case{83},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\s3.mat');AA=AA.d ;KK=15;    
case{84},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\s4.mat');AA=AA.d ;KK=15;    
case{85},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\dim512.txt');KK=max(AA(:,end));
case{86},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\a2.txt');KK=35;
case{87},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\a3.txt');KK=50;    
    
case{10},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\birch1-lab.txt');KK=100; % pdist出问题，距离数组太大
case{11},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\magic04.lab.txt');KK=14;
case{12},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\large2d\Skin_NonSkin.txt');KK=14; % pdist出问题，距离数组太大

case{13},AA=load('MLL1.mat');AA=AA.MLL1;KK=3;
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

case{30},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\parkinsons1.txt');KK=2;
case{31},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\balance-scale.txt');KK=3;
case{32},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\secom.txt');KK=2;
case{33},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\Internet Advertisements.txt');KK=max(AA(end,:));
case{34},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\Phishing Websites.txt');KK=max(AA(end,:));
case{35},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\Online Shoppers Purchasing Intention .txt');KK=max(AA(end,:)); 
case{36},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\pendigits.txt');KK=10;
case{37},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\NEWuci\RingNorm.txt');KK=2;  
case{38},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\uci\diabetes2.mat');KK=2; AA=AA.d; 
case{39},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\glass.data');KK=7;     
case{46},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\waveform(noise).mat');KK=3;AA=AA.d;  
end
%% 数据列归一化 
B1=AA(:,1:end-1); 
% B1=AA(:,:); 
%原始dpc整体都没有归一化
B= libsvmscale(B1,0,1);  % 数据归一化。 svm_scale是以列作为缩放的，对每种特征进行缩放
% B = B1 ;
[rows,dim]=size(AA);
A=[B,AA(:,end)];
LAB=AA(:,end);
%% 显示原始结果 
%数据格式为三列，前两列是二维数据，最后一列是类别  x,y,c  
%最多显示7中类别的聚类
%if( dim==3)  %改维数
   ShowClusterA(A,'origin graphic')

%DPC密度峰聚类
%ret=FADPC(A,B,KK);
%ret=FADPC_H(B,KK);  
%ret=FADPC_DIV(B,KK);
ret=DBSCAN(B,R,MinPts);

%  ret=DPC_t(B,KK);

%%%%%%    
%% 画图
F=[B,ret.cl'];
%if( dim==3)
ShowClusterA(F,' ');   

%画中心
% for i=1:length(ret.icl)
%    ccl=ret.icl(i);
%    scatter(B(ccl,1),B(ccl,2),100,'kh','MarkerFaceColor','w');%黑边六角星
%    hold on;
% end

% LB=ret.cl';
% critcurr1 = evalclusters(B,LB,'DaviesBouldin');
% disp(critcurr1) ;
% AMI = AMI(AA(:,end),ret.cl');
% [AMI,NMI] = NMI_AMI(AA(:,end)',ret.cl) ;
ARI = Library.GetAri(AA(:,end),ret.cl');
FMI = Library.GetFmi(AA(:,end),ret.cl');
% disp(NMI)
% ev = [AMI,ARI,FMI] ;
ev = [ARI,FMI] ;
disp(ev)




