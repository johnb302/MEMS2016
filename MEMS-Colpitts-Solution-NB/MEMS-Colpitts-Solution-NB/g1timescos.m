function [g1timescosout] = g1timescos(phi0,x,params)
% the function g1, representing the nondimensionalized transistor
U = x(1);
V = 0;
G = x(2);
H = x(3);
X = x(4);
Y = x(5);
mu = params.mu;
a1 = params.a1;
a2 = params.a2;
ep = params.ep;
gamma = params.gamma;
a0 = params.a0cur;
u0 = params.u0;

%equation34
v1mu1 = (V - U) + (G - X)*cos(phi0) + (H - Y)*sin(phi0);

% equation 28
cond = (ep*v1mu1 >= u0 - 1);
g1timescosout = (-(~cond).*ones(size(phi0)).*gamma./ep.*(1-u0)^2 + ...
    (cond).*gamma.*v1mu1.*(2.*(1-u0)+ep.*v1mu1)).*cos(phi0);

end

