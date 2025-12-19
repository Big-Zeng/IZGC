function z=H(dc,B)
[n,~]=size(B);

for i=1:n
   fa(i)=0;
   for j=1:n
       if i == j
           fa(i)=fa(i);
       else
      fa(i)=fa(i)+exp((norm(B(i,:)-B(j,:))/dc)^2);
       end
   end
end
sum_fa=sum(fa);
h=0;
for i=1:n
    h=h-(fa(i)/sum_fa)*log(fa(i)/sum_fa);
end
 % z=abs(1-h);
%z=h;
 %z=-h;
% 
 fa_max=max(fa);
 fa_min=min(fa);
 fa_mean=mean(fa);
 z=(fa_max-fa_min)/2-fa_mean;
%  %z=(fa_max-fa_min);
%z=fa_max-fa_mean;
%z=1/(1+z);
%z=fa_mean/(fa_max-fa_min);
 z=-z;

% for i=1:n
%    fa(i)=0;
%    for j=1:n
%        if i == j
%            fa(i)=fa(i);
%        else
%       fa(i)=fa(i)+exp((norm(B(i,:)-B(j,:))/dc)^2);
%        end
%    end
% end
%  fa=sort(fa);
%  s=0;
%  for i=1:n-1
%      s=s+fa(i+1)-fa(i);
%  end
%  z=-s;

end