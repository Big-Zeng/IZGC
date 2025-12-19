% clc;
clear;
close all;
% my DPC code
num = 3            ;

percent= 3.8          ; 

cores = 2           ; 
switch  num
case{1},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\flame.txt');KK=2;
case{2},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Aggregation.txt');KK=7;   
case{3},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\pathbased.txt');KK=3;
case{4},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\spiral.txt');KK=3;
case{5},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\jain.txt');KK=2;
case{6},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\Compound.txt');KK=6;%
case{7},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\R15.txt');KK=15;
case{8},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\D31.txt');KK=31;

case{91},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\10-19dataset\shape\sizes53.mat');KK=4;AA=AA.AA;    
case{92},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\cheng_data\artificial\five_cluster.mat');KK=5;AA=AA.d;        
case{93},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\sn3.mat');KK=3;AA=AA.d;  
case{94},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\creat\ring.txt');KK=2;    
case{95},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\creat\Zigzag.txt');KK=3;  

case{96},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\t4.8k.txt');KK=3;
    
case{100},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\ClusteringDataset\Realworld\OlivettiFaces\OlivettiFaces.mat');KK=40;AA=AA.d;     
case{101},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\DADCdatasets\Heartshaped.csv');KK=3; 
case{102},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\DADCdatasets\ED_Hexagon.csv');KK=2;     
case{103},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\DADCdatasets\G2.csv');KK=1;%一个簇
case{104},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\DADCdatasets\G50.csv');KK=4;%没类标    
case{105},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\DenMune\Mouse.txt');KK=4;     
case{106},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\DenMune\Very Density.txt');KK=3;  
    
case{111},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\shape\creat\Parabolic.txt');KK=2;
case{112},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\DPC一些改进算法程序\DPC-DBFN-master\datasets\fish.data');KK=7;
case{113},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\DPC一些改进算法程序\DPC-DBFN-master\datasets\ring3d.mat');KK=2;AA=AA.d;
    
    
case{50},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\circle3.mat'); KK=3;AA = AA.d;      
case{51},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\cth3.mat'); KK=4;AA = AA.d;
case{52},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\db13.mat'); KK=4;AA = AA.d;    
case{53},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\jain23.mat'); KK=2;AA = AA.d;     
case{54},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\newDatasets\line4.mat'); KK=4;AA = AA.d; %dpc达到1     
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
    
case{40},AA=xlsread('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\sonar.xlsx');KK=2;    
case{41},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\bupa(Liver).data');KK=2;
case{42},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\cmc(contraceptive).data');KK=3;    
case{43},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\haberman.data');KK=2;    
case{44},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\hayes-roth.data');KK=3;    
case{45},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\SPECTFincorrect.mat');KK=44;AA=AA.d; 
case{46},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\waveform(noise).mat');KK=3;AA=AA.d; 
case{47},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\SCADI\SCADI-Dataset\SCADI.csv');KK=7;
case{48},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\letter\letter.mat');KK=26;AA=AA.d;
case{200},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\new414\leuk.mat');KK=3;AA=AA.d;
case{201},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\new414\heart.mat');KK=2;AA=AA.d;
case{202},AA=load('C:\Users\w\Desktop\DPC参考文献\DPC\dataset\uci\newUCI\new414\vertebral_column_data\Vertebral.txt');KK=3;     
end
%% 数据列归一化 
B1=AA(:,1:end-1); 
%原始dpc整体都没有归一化
B= libsvmscale(B1,0,1);  %svm_scale是以列作为缩放的，对每种特征进行缩放。列归一化损失少量精度
% B=  mapminmax(B1,0,1) ;%行归一化损失大量精度
% B=B1 ;
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

% dist = pdist2(data,data);%直接算出 对角线为0的距离矩阵
% disp(data);
% disp(B);
%% 算局部密度
% 确定 dc  
[~,N] = size(dist_row) ;   % N = m*(m-1) / 2
% percent=  0.9  ;  
position=round(N*percent/100); %% round 是一个四舍五入函数  
sda=sort(dist_row); %% 对所有距离值作升序排列  
dc=sda(position);
if(dc<0.00000000001)
  dc=min(sda(sda~=0));%取sda中的最小距离作为dc
end

   %% 将每个数据点的 rho 值初始化为零  
    ND = size(B,1);
   rho=zeros(1,ND);
     for i=1:ND  
       rho(i)=0.;  
     end  

switch cores  
    case{1}
    % "Cut off" kernel  dc截断密度
    for i=1:ND-1  
     for j=i+1:ND  
       if (dist(i,j)<dc)  
          rho(i)=rho(i)+1.;  
          rho(j)=rho(j)+1.;  
       end  
     end  
    end  
    case{2}    
    % Gaussian kernel  
    for i=1:ND-1  
       for j=i+1:ND  
          rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));  
          rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));  
       end  
    end 
end
%% 算相对距离

    % 先求矩阵列最大值，再求最大值，最后得到所有距离值中的最大值      
    maxd=max(max(dist));   

    % 将 rho 按降序排列，ordrho 保持序           
    [rho_sorted,ordrho]=sort(rho,'descend');  

    % 处理 rho 值最大的数据点  
    delta(ordrho(1))=-1.;  
    nneigh(ordrho(1))=0;  

    %% 生成 delta 和 nneigh 数组  算相对距离
    for ii=2:ND  
       delta(ordrho(ii))=maxd;  
       for jj=1:ii-1  
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))  
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));  
            
            nneigh(ordrho(ii))=ordrho(jj);  
            %% nneigh放的是父节点
            %% 记录 rho 值更大的数据点中与 ordrho(ii) 距离最近的点的编号 ordrho(jj)  
         end  
       end  
    end
    
%% 找类簇中心    
    % 生成 rho 值最大数据点的 delta 值
    delta(ordrho(1))=max(delta(:));  
% 数据归一化
rho=mapminmax(rho,0,1);
delta=mapminmax(delta,0,1);

    gamma=zeros(1,ND);
    for i=1:ND                   %ND为数据总行数   
        gamma(i)=rho(i)*delta(i);
    end  
    %对gamma进行排序，然后选择突变（跳变）的那些。在这里只选择前 7个最大的    （在ShowClusterA画图函数中，三位二进制共有7种颜色）
    [gamma_sorted,gamma_order]=sort(gamma,'descend');
       
  %% gamma传参给 cccc，为选择聚类中心做准备 
%     std_val=std(gamma_sorted);
%     class_num=length(find(gamma_sorted>3*std_val));
     class_num=KK;
     cccc=gamma_order(1:class_num);  %%  KK决定聚类中心个数
    
    
%% 利用 rho 和 delta 画出一个所谓的“决策图”  
    figure;
    tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');  
    title (' ','FontSize',15.0)  
    xlabel ('\rho')  
    ylabel ('\delta')

%     figure;
%     tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');  
%     hold on ;
%     title (' ','FontSize',15.0)  
%     xlabel ('\rho')  
%     ylabel ('\delta')
%     for i=1:length(rho)
%     scatter(rho(i),delta(i),25,'ko','MarkerFaceColor','k');%
%     hold on;
%     end

%% 画gamma降序图
%     figure;
%     [gamma_de,~] = sort(gamma,'descend') ; 
%     ttt = plot( (1:length(gamma)), gamma_de(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k') ;
%     hold on ;
%     title (' ','FontSize',15.0)  
%     xlabel ('n')  
%     ylabel ('\gamma')
    
%%% 分开画   
%     for j=1:length(gamma)
%     scatter( j, gamma_de(j),25,'ko','MarkerFaceColor','k');%
%     hold on;
%     end
      
%% 样本点分配

    % 初始化 cluster 个数  
%     NCLUST=0;  

    %% cl 为归属标志数组，cl(i)=j 表示第 i 号数据点归属于第 j 个 cluster  
    % 先统一将 cl 初始化为 -1  
    cl=zeros(1,ND);
    for i=1:ND  
      cl(i)=-1;  
    end  


%% cccc数组中为 类簇中心点，最大的在第一个
% cl中非类簇中心点都是-1
    [~,NCLUST]=size(cccc);
    icl = zeros(1,NCLUST) ;
    for i=1:NCLUST
        cl(cccc(i))=i;
        icl(i)=cccc(i);
    end
    
%% 将其他数据点归类 (assignation)  
    for i=1:ND  
      if (cl(ordrho(i))==-1)  
        cl(ordrho(i))=cl(nneigh(ordrho(i)));  
      end  
    end  
 
   
%%  去除噪声  (dc修改为dc1)

%     %% 定义dc1
% ks = 20   ;
% [dist1Sort,dist1Order]=sort(dist,2); %每行排序
% 
% temp = dist1Sort(: , 2:ks) ;
% dc1 =  sum(mean(sum(temp))) / rows ;

dc1 = dc ;

    % 处理光晕点，halo这段代码应该移到 if (NCLUST>1) 内去比较好吧  
     halo=zeros(1,ND);
    for i=1:ND  
      halo(i)=cl(i);  
    end  

    if (NCLUST>1)  

      % 初始化数组 bord_rho 为 0,每个 cluster 定义一个 bord_rho 值  
       bord_rho=zeros(1,NCLUST);
      for i=1:NCLUST  
        bord_rho(i)=0.;  
      end  

      % 获取每一个 cluster 中平均密度的一个界 bord_rho ，选出的是两个密度最大点的均值 
      for i=1:ND-1  
        for j=i+1:ND  
          %% 距离足够小但不属于同一个 cluster 的 i 和 j  
          if ((cl(i)~=cl(j))&& (dist(i,j) <= dc1 ))  
            rho_aver=(rho(i)+rho(j))/2.; %% 取 i,j 两点的平均局部密度  
            if (rho_aver>bord_rho(cl(i)))   
              bord_rho(cl(i))=rho_aver;  
            end  
            if (rho_aver>bord_rho(cl(j)))   
              bord_rho(cl(j))=rho_aver;  
            end  
          end  
        end  
      end  

      %% halo 值为 0 表示为 outlier  
      for i=1:ND  
        if (rho(i)<bord_rho(cl(i)))  
          halo(i)=0;  
        end  
      end  

    end  

    %% 逐一处理每个 cluster  
    for i=1:NCLUST  
      nc=0; %% 用于累计当前 cluster 中数据点的个数  
      nh=0; %% 用于累计当前 cluster 中核心数据点的个数  
      for j=1:ND  
        if (cl(j)==i)   
          nc=nc+1;  
        end  
        if (halo(j)==i)   
          nh=nh+1;  
        end  
      end  
%暂时不输出
%        fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);  

    end
%% 标注噪声

% cl = halo ;    
    
%% 画图
F=[B,cl'];
%if( dim==3)
ShowClusterA(F,' ');   

%%
% 画中心
for i=1:length(icl)
   ccl=icl(i);
   scatter(B(ccl,1),B(ccl,2),100,'kp','MarkerFaceColor','w');%黑边五角星
%    scatter(B(ccl,1),B(ccl,2),'p','k');%黑边五角星
   hold on;
end
 
LB=cl';
% critcurr1 = evalclusters(B,LB,'DaviesBouldin');
% disp(critcurr1) ;
% [RI,ARI,NMI]=evolution(A,F,KK);
% [AMI,NMI] = NMI_AMI(AA(:,end)',cl) ;
AMI = AMI(AA(:,end),cl');
ARI = Library.GetAri(AA(:,end),cl');
FMI = Library.GetFmi(AA(:,end),cl');
% disp(RI)
% disp(NMI)
% FM(F,A) = Fmeasure(AA(:,end),cl');
ev = [AMI ,ARI,FMI] ;
% ev = [AMI ,NMI,F] ;
% ev = [AMI ,NMI,FMI] ;
disp(ev)

% [silh,h] = silhouette(B1,cl,'sqeuclidean');
% mean(silh)
