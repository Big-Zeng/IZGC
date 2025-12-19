function [ ret ] = FNDPC(kexi,A,KK )%,class_num,k)
%CFSFDP 输入距离矩阵返回最大聚类的所有点
%   此处显示详细说明
%     xx=load('example_distances.dat');
%     clear all  
%     close all  
%     disp('The only input needed is a distance matrix file')  
%     disp('The format of this file should be: ')  
%     disp('Column 1: id of element i')  
%     disp('Column 2: id of element j')  
%     disp('Column 3: dist(i,j)')  
%% 作距离矩阵
B=pdist2(A,A,'minkowski',2);

 [row_b,col_b]=size(B);
%  clear distance;
 xx=zeros(row_b*(row_b-1)/2,3);%大小为m*(m-1)/2
 index=1;
 for i=1:row_b
     for j=i+1:col_b
         xx(index,1)=i;
         xx(index,2)=j;
         xx(index,3)=B(i,j);
         index=index+1;
     end
 end
 %% 检查是否出错
    ND=max(xx(:,2));  
    NL=max(xx(:,1));  
    if (NL>ND)  
      ND=NL;  %% 确保 DN 取为第一二列最大值中的较大者，并将其作为数据点总数  
    end  

    N=size(xx,1); %% xx 第一个维度的长度，相当于文件的行数（即距离的总个数）  

    %% 初始化为零    
    %使用矩阵初始化，提高程序运行速度
   dist=zeros(ND,ND);
     for i=1:ND  
      for j=1:ND  
         dist(i,j)=0;  
       end  
     end  

    %% 利用 xx 为 dist 数组赋值，注意输入只存了 0.5*DN(DN-1) 个值，这里将其补成了满矩阵  
    %% 这里不考虑对角线元素  
    for i=1:N  
      ii=xx(i,1);  
      jj=xx(i,2);  
      dist(ii,jj)=xx(i,3);  
      dist(jj,ii)=xx(i,3);  
    end  
    
    %% 确定 dc  
  % kexi=0.01;
   dmax=max(xx(:,3));
   dmin=min(xx(:,3));
   dmean=mean(xx(:,3));
   dc=kexi*dmax;

    %% 计算局部密度 rho (利用 Gaussian 核)  

%     fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);  

    %% 将每个数据点的 rho 值初始化为零  
    %rho=zeros(ND);
     for i=1:ND  
       rho(i)=0.;  
     end  
%     for i=1:ND
%         dist1=sort(dist(i,:));
%         dc=sum(dist1(1:10))/10;
%         for j=1:ND
%             if i~=j
%                 rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%             end
%         end
%     end
    %ksdensity核密度估计
%     hmise=mean(1.06*std(B,0,1)*row_b^(-0.2));
%     p = kde(A', hmise ); 
   % p=kde(A', 'rot');
  %  rho=evaluate(p, A');
    % Gaussian kernel  
%     for i=1:ND-1  
%        for j=i+1:ND  
%           rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));  
%           rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));  
%        end  
%     end  
%%%%%%%%%%%%%%%%%%%
    for i=1:ND-1  
       for j=i+1:ND 
           if (dist(i,j)<dc) 
            rho(i)=rho(i)+(1-dist(i,j)/dc);
            rho(j)=rho(j)+(1-dist(i,j)/dc); 
           end 
       end  
    end  
%     for i=1:ND-1  
%        for j=i+1:ND 
%            if (dist(i,j)<dc) 
%             rho(i)=rho(i)+cos((dist(i,j)/dc));
%             rho(j)=rho(j)+cos((dist(i,j)/dc)); 
%            end 
%        end  
%     end  
%             figure
%         subplot(1,2,1);plot(rho);title('origin rho');    
%         subplot(1,2,2);plot(rho2);title('kde rho'); 

    % "Cut off" kernel  
%    for i=1:ND-1  
%      for j=i+1:ND  
%        if (dist(i,j)<dc)  
%           rho(i)=rho(i)+1.;  
%           rho(j)=rho(j)+1.;  
%        end  
%      end  
%    end  

    %% 先求矩阵列最大值，再求最大值，最后得到所有距离值中的最大值  
    maxd=max(max(dist));   

    %% 将 rho 按降序排列，ordrho 保持序  
    [rho_sorted,ordrho]=sort(rho,'descend');  

    %% 处理 rho 值最大的数据点  
    delta(ordrho(1))=-1.;  
    nneigh(ordrho(1))=0;  

    %% 生成 delta 和 nneigh 数组  
    for ii=2:ND  
       delta(ordrho(ii))=maxd;  
       for jj=1:ii-1  
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))  
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));  
            %这个值后面是不是可以用来归类？？？
            nneigh(ordrho(ii))=ordrho(jj);   
            %% 记录 rho 值更大的数据点中与 ordrho(ii) 距离最近的点的编号 ordrho(jj)  
         end  
       end  
    end  

    %% 生成 rho 值最大数据点的 delta 值  
    delta(ordrho(1))=max(delta(:));  

    %% 决策图  

%     disp('Generated file:DECISION GRAPH')   
%     disp('column 1:Density')  
%     disp('column 2:Delta')  

%     fid = fopen('DECISION_GRAPH', 'w');  
%     for i=1:ND  
%        fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));  
%     end  

%% 数据归一化
rho=mapminmax(rho,0,1);
delta=mapminmax(delta,0,1);
   
%% ind 和 gamma 选择密度峰值点，在后面并没有用到  
     ind=zeros(ND);
     gamma=zeros(ND);
    for i=1:ND  
      ind(i)=i;   
      gamma(i)=rho(i)*delta(i);  
    end  

    %对gamma进行排序，然后选择突变的那些。在这里只选择前7个最大的
    [gamma_sorted,gamma_order]=sort(gamma,'descend');
    class_num=KK;
    cccc=gamma_order(1:class_num);
    
    %% 初始化 cluster 个数  
    NCLUST=0;  

    %% cl 为归属标志数组，cl(i)=j 表示第 i 号数据点归属于第 j 个 cluster  
    %% 先统一将 cl 初始化为 -1  
%     cl=ones(ND)*(-1);
    for i=1:ND  
      cl(i)=-1;  
    end  

    %直接利用cccc数组获取中心点，最大的在第一个,其它的非聚类中心点都是-1
    [~,NCLUST]=size(cccc);
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
    %% 直接返回中心点及cluster点
    ret.icl=icl;
    ret.cl=cl;
    ret.gamma_sorted=gamma_sorted;
%    ret.halo=halo;
    ret.delta=delta;
    ret.rho=rho;
    ret.nneigh=nneigh;
    ret.dc=dc;
           
end

