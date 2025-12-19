   %%%%%FA优化dc%%%%%%%%%%%%%%%%%%%%
   function [ret]=FADPC_gd(A,B,KK,k)
   C=pdist2(B,B,'minkowski',2);
    [row_b,col_b]=size(C);
%  clear distance;
 xx=zeros(row_b*(row_b-1)/2,3);%大小为m*(m-1)/2
 index=1;
 for i=1:row_b
     for j=i+1:col_b
         xx(index,1)=i;
         xx(index,2)=j;
         xx(index,3)=C(i,j);
         index=index+1;
     end
 end
 
dcmin = min(xx(:,3));     % Lower Bound of Decision Variables
dcmax = max(xx(:,3));     % Upper Bound of Decision Variables
%% Firefly Algorithm Parameters
MaxIt=10;         % Maximum Number of Iterations
nPop=10;            % Number of Fireflies (Swarm Size)
gamma=1;            % Light Absorption Coefficient
beta0=1;            % Attraction Coefficient Base Value
alpha=0.2;          % Mutation Coefficient
alpha_damp=0.98;    % Mutation Coefficient Damping Ratio
delta=0.5*(dcmax-dcmin);     % Uniform Mutation Range
m=2;


%% Initialization

%% Initialization

% Empty Firefly Structure
firefly.Position=[];
firefly.Cost=[];
% Initialize Population Array
pop=repmat(firefly,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Fireflies
for i=1:nPop
   pop(i).Position=unifrnd(dcmin,dcmax);
   
   pop(i).Cost=RI_gd(pop(i).Position,A,B,KK,k); % 调RI_gd函数 （A为原始数据，B为去类标、归一化数据）
%    pop(i).Cost=Sil_gd(pop(i).Position,A,B,KK,k);
%    pop(i).Cost=H_gd(pop(i).Position,A,B,KK,k);
   
   if pop(i).Cost<=BestSol.Cost
       BestSol=pop(i);
   end
end

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% Firefly Algorithm Main Loop

for it=1:MaxIt
    
    newpop=repmat(firefly,nPop,1);
    for i=1:nPop
        newpop(i).Cost = inf;
        for j=1:nPop
              e=delta*unifrnd(-1,+1);
            if pop(j).Cost < pop(i).Cost
                rij=norm(pop(i).Position-pop(j).Position);
                beta=beta0*exp(-gamma*rij^m);
               % beta=betamin+(beta0-betamin)*exp(-gamma*rij^2);
                %e=delta*randn(VarSize);        
                newsol.Position = pop(i).Position ...
                                + beta.*(pop(j).Position-pop(i).Position) ...
                                + alpha*e;
            else
                newsol.Position=pop(i).Position +alpha*e;
            end
                newsol.Position=max(newsol.Position,dcmin);
                newsol.Position=min(newsol.Position,dcmax);
                
               newsol.Cost=RI(newsol.Position,A,B,KK);
                
                if newsol.Cost <= newpop(i).Cost
                    newpop(i) = newsol;
                    if newpop(i).Cost<=BestSol.Cost
                        BestSol=newpop(i);
                    end
                end
                
        end
    end
  
    
    % Merge
    pop=[pop
         newpop];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    Bestdc(it)=BestSol.Position;
    
     % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;
    %alpha = (1/9000)^(1/it)*alpha;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) ':Best dc=' num2str(Bestdc(it))]);
end
[ret]=new24_gdFA(BestSol.Position,B,KK,k); %调子函数
