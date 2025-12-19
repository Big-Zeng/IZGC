function [db,mu,Number]= DB(DATA,cl)
% [~,DATA_length]=size(DATA);
K=max(cl);
for i=1:K
    %求簇聚类中心
    num=find(cl==i);
    [a,~]=size(DATA(num,:));
    Number(i)=a;
    if a==1
        mu(i,:)=DATA(num,:);
    else
        mu(i,:)=mean(DATA(num,:));
    end
    %求簇内平均距离
    B=pdist2(mu(i,:),DATA(num,:),'minkowski',2);
    C(i)=mean(B);
end
DB=zeros(K,K);
for i=1:K-1
    for j=i+1:K
        DB(i,j)=(C(i)+C(j))/sqrt(sum((mu(i,:)-mu(j,:)).^2));
    end
end
DB=DB+DB';
db=sum(max(DB))/K;