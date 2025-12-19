function  ShowClusterA( A,ctitle )
%SHOWCLUSTERA 此处显示有关此函数的摘要
%数据格式为三列，前两列是二维数据，最后一列是类别  x,y,c  
%最多显示7中类别的聚类，聚的7个类

% colors=['r','g','b','y','m','c','k'];
[row,col]=size(A);
pointStyles=['+', '*', '.', 'x','o','s','d','p','h','>'];
L=unique(A(:,col));
N=length(L);
lineStyles = linspecer(N);       %创建N种颜色用于不同线条的绘制   8-1 种颜色
lineStyles=[[1,1,1];lineStyles];
figure;
for i=1:N
    ir = find(A(:,col)==L(i,1));         % 返回行索引  找到同一个类簇的所有点，一次画一个簇
    if(~isempty(ir))
        if col>3                          %         填充颜色                           边缘颜色
            scatter3(A(ir,1),A(ir,2),A(ir,3),'MarkerFaceColor',lineStyles(i+1,:),'MarkerEdgeColor',lineStyles(i+1,:));%,'Marker','.'
        else
            scatter(A(ir,1),A(ir,2),20,'MarkerFaceColor',lineStyles(i+1,:),'MarkerEdgeColor',lineStyles(i+1,:));          
%             scatter(A(ir,1),A(ir,2),'MarkerFaceColor',lineStyles(i+1,:),'MarkerEdgeColor',lineStyles(i+1,:));   
        end
        hold on
    end
   
end
% hold off
title(ctitle);

end

