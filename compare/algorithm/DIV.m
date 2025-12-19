function z=DIV(dc,B,c)
ret=DPC(dc,B,c);
F=[B,ret.cl'];
[n,d]=size(F);
%%%%%%%%%%%%%%%%%%%%%%%%%%评价函数程序
%%%%% 原始数据集类元素统计
for i=1:c
     b(i).jh=[];
end
 for i=1:n
     for j=1:c
     if F(i,d)==j
         b(j).jh=[b(j).jh;F(i,1:d-1)];
     end
     end
 end
 nn=1;
 Dist1=[];
 Dist2=[];
 for i=1:c-1
     c=b(i).jh;
     num1=size(c,1);
     for j=i+1:c
         d=b(j).jh;
         num2=size(d,1);
         for k1=1:num1
             for k2=1:num2
                 Dist1(nn)=norm(c(num1)-d(num2));
                 nn=nn+1;
             end
         end
     end
 end
 F1=min(Dist1);
 for i=1:c
     e=b(i).jh;
     num3=size(e,1);
     for j=1:num3-1
         Dist2(j)=norm(e(j+1)-e(j));
     end
 end
 F2=max(Dist2);
 z=F1/F2;
 z=-z;
end