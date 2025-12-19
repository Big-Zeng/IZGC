   %%%%%FA优化dc%%%%%%%%%%%%%%%%%%%%
   function [ret]=FADPC_H(B,KK)
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
% dcmin = 0.13;     % Lower Bound of Decision Variables
% dcmax = 0.14;     % Upper Bound of Decision Variables
%% Firefly Algorithm Parameters
MaxIt=10;         % Maximum Number of Iterations
nPop=10;            % Number of Fireflies (Swarm Size)
gamma=1;            % Light Absorption Coefficient
beta0=1;            % Attraction Coefficient Base Value
betamin=0.2;
alpha=0.2;          % Mutation Coefficient
alpha_damp=0.98;    % Mutation Coefficient Damping Ratio
delta=0.05*(dcmax-dcmin);     % Uniform Mutation Range
m=2;
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
   pop(i).Cost=H(pop(i).Position,B);
   
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
            if pop(j).Cost < pop(i).Cost
                rij=norm(pop(i).Position-pop(j).Position);
                beta=beta0*exp(-gamma*rij^m);
                %beta=betamin+(beta0-betamin)*exp(-gamma*rij^2);
                e=delta*unifrnd(-1,+1);
                %e=delta*randn(VarSize);
                
                newsol.Position = pop(i).Position ...
                                + beta*(pop(j).Position-pop(i).Position) ...
                                + alpha*e;
            
                newsol.Position=max(newsol.Position,dcmin);
                newsol.Position=min(newsol.Position,dcmax);
                
                newsol.Cost=H(newsol.Position,B);
                
                if newsol.Cost <= newpop(i).Cost
                    newpop(i) = newsol;
                    if newpop(i).Cost<=BestSol.Cost
                        BestSol=newpop(i);
                    end
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
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) ':Best dc=' num2str(BestSol.Position)]);
    
    % Damp Mutation Coefficient
    alpha = alpha*alpha_damp;
    %alpha = ((1/9000)^(1/it))*alpha; 
    
end
[ret]=DPC(BestSol.Position,B,KK);
