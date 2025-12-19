% 完成版
% DPC-FNWSN 算法  （PR论文）

% clc;
clear;
close all;

num = 2   ;

k =  11    ;        %1-12;2-30;3-4;5-5；14-20；
K=k;
% percent=  0.1    ; 
switch  num
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
case{57},AA=load('F:\科研\论文\分区论文\distrubuteArea\compare\10-19dataset\shape\LineBlobs2.mat'); KK=3;AA = AA.d; 
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
%% 数据列归一化 
B1=AA(:,1:end-1); 
%原始dpc整体都没有归一化
B= libsvmscale(B1,0,1);  %svm_scale是以列作为缩放的，对每种特征进行缩放。列归一化损失少量精度
[rows,dim]=size(AA);
A=[B,AA(:,end)];
LAB=AA(:,end);
%% 显示原始结果 
%数据格式为三列，前两列是二维数据，最后一列是类别  x,y,c  
%最多显示7中类别的聚类
%if( dim==3)  %改维数
   ShowClusterA(A,'origin graphic')

%% 算所有点的距离
%Calculate dist1
data = B;                   %除去最后一行标签，归一化后的数据
dist_row = pdist(data);     %pdist（data）计算 data 中 各对行向量 的相互距离 存储为一个行向量
dist=squareform(dist_row);  % squareform转成对 角线为零的 距离矩阵
   %% 将每个数据点的 rho 值初始化为零  
    ND = size(B,1);
   rho=zeros(1,ND);
     for i=1:ND  
       rho(i)=0.;  
     end  
%% 隶属模糊核

[dist_sort , dist_order ]=sort(dist,2) ;           %每行排序
% tempw = sum( dist_sort(: ,2:k ) ,2 )  ;
tempw = dist_sort(: ,: )   ;

tk1 = sum( dist_sort(: ,: ) ,2 ) ;
w = sum(tk1) /(0.5*(ND-1)*ND)    ;   %全局距离

f1 =  sqrt( sum(sum(( dist - w).^2 )) / (ND^2) )          ; % 全局方差

tempk1 = 1./ ( dist_sort(: ,2:k ) +1  ).^2 ;
tempk2 =   1  ./ ( dist_sort(: ,k+1:end )*1 + 1   ).^2 ;    %

tempke = exp( -(dist_sort(: ,2:k ).*dist_sort(: ,2:k ) ./ 1 ) )  ;
tempkf = exp( -(dist_sort(: ,k+1:end ).*dist_sort(: ,k+1:end )* f1^2 ./ 1  ) )  ;

rho1 = sum( tempk1 .* tempke ,2 )'/(k-1) ;
rho2 = sum( tempk2 .* tempkf ,2 )'/(ND-k+1) ;
rho = rho1 + rho2 ;

%% 

temk1 =   1./ ( dist_sort(: ,: ) +1  ).^2 ;
temk2 =   1./ ( dist_sort(: ,: ) + 1 ).^2 ;    %求得每个点前几个近邻的隶属度

temke = exp( -(dist_sort(: ,: ).*dist_sort(: ,:) ) )  ;
temkf = exp( -(dist_sort(: ,:).*dist_sort(: ,:)* f1^2   ) )  ;

u_knn = temk1 .* temke ;
u_not = temk2 .* temkf ;

u_knn(:,[1,k+1:end]) = 0 ;
u_not(:,1:k ) = 0 ;
uu = u_knn + u_not ;

u = zeros(ND,ND);

for i = 1:ND
    u(i,dist_order(i,:)) = uu(i,:) ; % 原序
       
end


%%  相互近邻相似度
	Pik_sorted=zeros(ND,ND);
    Pik_order=zeros(ND,ND);
    for i=1:ND
        [Pik_sorted(:,i),Pik_order(:,i)]=sort(dist(:,i));%sort列排序
    end
%%%    
    dist_1=zeros(ND,ND);
    A2=zeros(ND,ND);
    
    for i=1:ND-1
        for j=i+1:ND
            if Pik_sorted(K+1,i)+Pik_sorted(K+1,j)>=dist(i,j)
                c=intersect(Pik_order(2:K+1,i),Pik_order(2:K+1,j));
                A2(i,j)=size(c,1);      %共享近邻snn(i,j)的相似度
            end
        end
    end
    A2=(A2+A2');
    A2=A2/(K);
    DDt=A2; 
   
    DD = DDt.* u ;    
%     DD = DDt + u ;      
    
    DD(isnan(DD))=0;

%%
    %%% 先求矩阵列最大值，再求最大值，最后得到所有距离值中的最大值  
    maxd=max(max(dist));   
    %%% 将 rho 按降序排列，ordrho 保持序  
    [~,ordrho]=sort(rho,'descend');  
    %%% 处理 rho 值最大的数据点  
    delta(ordrho(1))=-1.;  
    nneigh(ordrho(1))=0;  
    %% 生成 delta 和 nneigh 数组  
    for ii=2:ND  
       delta(ordrho(ii))=maxd;  
       for jj=1:ii-1  
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))  
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));  
            nneigh(ordrho(ii))=ordrho(jj);   
         end  
       end  
    end  

    delta(ordrho(1))=max(delta(:));    
%% 数据归一化
rho=mapminmax(rho,0,1);
delta=mapminmax(delta,0,1);
% plot(rho,delta,'*');
   
    ind=zeros(1,ND);
    gamma=zeros(1,ND);
    for i=1:ND  
      ind(i)=i;   
      gamma(i)=rho(i)*delta(i);  
    end
    
    [gamma_sorted,gamma_order]=sort(gamma,'descend');
   
	class_num=KK;
	cccc=gamma_order(1:class_num);
    
    
%% 利用 rho 和 delta 画出一个所谓的“决策图”  
%     figure;
%     tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');  
%     hold on ;
%     title (' ','FontSize',15.0)  
%     xlabel ('\rho')  
%     ylabel ('\delta')
% %     for i=1:length(rho)
% %     scatter(rho(i),delta(i),25,'ko','MarkerFaceColor','k');%
% %     hold on;
% %     end

%% 画gamma降序图
%     figure;
%     [gamma_de,~] = sort(gamma,'descend') ; 
%     ttt = plot( (1:length(gamma)), gamma_de(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k') ;
%     hold on ;
%     title (' ','FontSize',15.0)  
%     xlabel ('n')  
%     ylabel ('\gamma')
%     
% %%% 分开画   
% %     for j=1:length(gamma)
% %     scatter( j, gamma_de(j),25,'ko','MarkerFaceColor','k');%
% %     hold on;
% %     end   
    
    
    
%% 相似度矩阵
    for i=1:ND
        DD(i,i)=0;
    end
    cl=ones(1,ND)*(-1);
    icl=zeros(1,KK);
	for i=1:KK
        cl(cccc(i))=i;
        icl(i)=cccc(i);
	end
	DD(cccc,:)=zeros(size(cccc,2),ND);

        while 1
            AAA=DD;
            c=find(cl>0);   %中心位置
            c1=max(max(DD(:,c)));%
            if c1==0
                break %跳出
            end
            while 1
                [a,b]=find(DD(:,c)==c1);    % a行下标，b列下标
                if cl(a(1))<0
                    cl(a(1))=cl(c(b(1)));
                    DD(a(1),:)=zeros(1,ND);
                    break
                else
                    DD(a(1),c(b(1)))=0;
                    AAA(a(1),c(b(1)))=0;
                end
            end
            if min(cl)==1
                break
            end
            if max(max(DD))==0
                break
            end
        end
%%%       
        if size(c,2)<ND-1
            for i=1:ND  
                if (cl(ordrho(i))==-1)  
                    cl(ordrho(i))=cl(nneigh(ordrho(i)));  
                end  
            end
        end
        
        icl=zeros(1,KK);
        for i=1:class_num
            icl(i)=cccc(i);
        end

%% 画图
F=[B,cl']   ;
%if( dim==3)
ShowClusterA(F,'  ')    ;   

%画中心

for i=1:length(icl)
   ccl=icl(i);
%    scatter(B(ccl,1),B(ccl,2),'p','k');%
   scatter(B(ccl,1),B(ccl,2),100,'kp','MarkerFaceColor','w');%黑边五角星
   hold on;
end
% LB=cl';
% critcurr1 = evalclusters(B,LB,'DaviesBouldin');
% disp(critcurr1) ;
% [RI,ARI,NMI]=evolution(A,F,KK);
% [AMI,NMI] = NMI_AMI(AA(:,end)',cl) ;
AMI = AMI(AA(:,end),cl');
ARI = Library.GetAri(AA(:,end),cl');
FMI = Library.GetFmi(AA(:,end),cl');
% disp(RI)
% disp(NMI)
ev = [AMI ,ARI,FMI] ;
disp(ev)
