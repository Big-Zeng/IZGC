function z=Sil(dc,B,c)
ret=DPC(dc,B,c);
F=[B,ret.cl'];
[n,d]=size(F);
num1=0;
num2=0;
for i=1:c
     aa(i).data=[];
 end
 for i=1:n
     for j=1:c
        if F(i,d)==j
         aa(j).data=[aa(j).data;F(i,1:d-1)];
        end
     end
 end  
 for i=1:n
     for j=1:n
     if F(i,d)==F(j,d)
         num1=num1+1;
         dist1(num1)=pdist2(F(i,1:d-1),F(j,1:d-1));
     end
     end
     for j=1:c
     if F(i,d)~=j
         num2=num2+1;
         dist=pdist2(F(i,1:d-1),aa(j).data);
         dist2(num2)=sum(dist)/size(aa(j).data,1);
     end
     end
     a(i)=sum(dist1)/(num1-1);
     b(i)=min(dist2);
  sil(i)=(b(i)-a(i))/max(a(i),b(i));
 end
 Sil=mean(sil);
 %z=-Sil;
 z=1/(1+Sil);
end
 

   
   
   
   
    

