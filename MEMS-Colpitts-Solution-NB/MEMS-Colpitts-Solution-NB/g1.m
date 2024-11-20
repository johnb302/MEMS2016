function [g1out] = g1(phi0,x,params)
% the function g1, representing the nondimensionalized transistor
U = x(1);
V = 0;
G = x(2);
H = x(3);
X = x(4);
Y = x(5);

ep = params.ep;
gamma = params.gamma;
u0 = params.u0;

%equation34
v1mu1 = (V - U) + (G - X)*cos(phi0) + (H - Y)*sin(phi0);
% equation 28
cond = (ep*v1mu1 >= u0 - 1);

g1out = -(~cond).*ones(size(phi0)).*gamma./ep.*(1-u0)^2 + ...
    (cond).*gamma.*v1mu1.*(2.*(1-u0)+ep.*v1mu1);

end