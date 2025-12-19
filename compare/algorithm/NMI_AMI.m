function [NMI, AMI] = NMI_AMI(X, Y)
%NMI_AMI return NMI, AMI
% MI: mutual information
% H; entropy
% NMI: normalized mutual infomation
% AMI: adjusted mutual infomation
% NMI(X, Y) = MI(X, Y) / F(H(X), H(Y))
% AMI(X, Y) = (MI(X, Y) - EMI(X, Y)) / (F(H(X) + H(Y)) - EMI(X, Y))
% F(x, y) is a function, can be "mean", "max", "geometric", "arithmetic"
% here we both use arithmetric

NMI = 2 * MI(X, Y) / (H(X) + H(Y));

AMI = (MI(X, Y) - EMI(X, Y)) / (1/2 * (H(X) + H(Y)) - EMI(X, Y));

end

function [res] = MI(X, Y)
%MI mutual infomation

n = length(X);
X_list = unique(X);
Y_list = unique(Y);
res = 0;
for x = X_list
    for y = Y_list
        loc_x = find(X == x);
        loc_y = find(Y == y);
        loc_xy = intersect(loc_x, loc_y);
        res = res + length(loc_xy) / n * log(length(loc_xy) / n / ((length(loc_x) / n) * (length(loc_y) / n)) + eps);
    end
end

end

function [res] = H(X)
%H information entropy

n = length(X);
X_list = unique(X);
res = 0;

for x = X_list
    loc = find(X == x);
    px = length(loc) / n;
    res = res - px * log(px);
end

end

function [res] = f(a, b)
% F calculate a! / b!
% sometimes a and b can be very large, hence, directly calculate a! or b! is not
% suitable; but maybe a-b is small; 
% a,b should both be positive integers

res = 1;
if a > b
    for i = b+1:a
        res = res * i;
    end
elseif a < b
    for i = a+1:b
        res = res / i;
    end
else
    res = 1;
end

end

function [res] = EMI(U, V)
% EMI expected mutual information, E[MI(X, Y)]

N = length(U);

U_list = unique(U);
V_list = unique(V);
R = length(U_list);
C = length(V_list);

M = zeros(R, C);
for i = 1:R
    for j = 1:C
        U_loc = find(U == U_list(i));
        V_loc = find(V == V_list(j));
        M(i, j) = length(intersect(U_loc, V_loc));
    end
end

a = sum(M, 2);
b = sum(M, 1);
res = 0;

for i = 1:R
    for j = 1:C
        for nij = max(a(i) + b(j) - N, 1):min(a(i), b(j))
            res = res + nij / N * log(N * nij / (a(i) * b(j)) + eps) * f(a(i), a(i) - nij) * f(b(j), b(j) - nij) * f(N - a(i), N) * f(N - b(j), N - a(i) - b(j) + nij) / factorial(nij);
        end
    end
end

end
