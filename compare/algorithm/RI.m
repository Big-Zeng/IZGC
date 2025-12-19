function z=RI(dc,A,B,c)
%%
ret=DPC(dc,B,c);
%%
F=[B,ret.cl'];
[n,d]=size(F);
%%%%%%%%%%%%%%%%%%%%%%%%%%评价函数程序
f11=0;
f00=0;
fz=0;
fm1=0;
fm2=0;
r1=0;
r2=0;
%%%%% 原始数据集类元素统计
for i=1:c
     a(i).jh=[];
     b(i).jh=[];
 end
 for i=1:n
     for j=1:c
     if A(i,d)==j
        a(j).jh=[a(j).jh;A(i,1:d-1)];
     end
     if F(i,d)==j
         b(j).jh=[b(j).jh;F(i,1:d-1)];
     end
     end
 end
 %%%%%%%%%%%%%%%构建混淆矩阵
 for i=1:c
     for j=1:c
         if size(a(j).jh,1)>0
            hx(i,j)=sum(ismember(a(i).jh,b(j).jh,'rows'));
            if hx(i,j)>=2
               f11=nchoosek(hx(i,j), 2)+f11;
            end
         else
             hx(i,j)=0;
         end
     end
 end
 for i=1:c
     for j=1:c
         f00=hx(i,j)*(sum(sum(hx))-sum(hx(i,:))-sum(hx(:,j))+hx(i,j))+f00;
     end
 end
 f00=f00/2;
 %%%%%%% Rand Index
 z=(f00+f11)/(n*(n-1)/2);
 z=-z;
end
