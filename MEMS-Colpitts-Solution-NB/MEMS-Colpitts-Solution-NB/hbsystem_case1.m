function [F] = hbsystem_case1(x,params)

% solve this if v1-u1 is not sufficiently large
mu = params.mu;
a1 = params.a1;
a2 = params.a2;
ep = params.ep;
gamma = params.gamma;
a0 = params.a0cur;
u0 = params.u0;

% corresponding U, V, G, H, X, Y
U = x(1);
V = 0;
G = x(2);
H = x(3);
X = x(4);
Y = x(5);

% note that F is a function of the harmonic balance coefficients (fourier
% coefficients) as well as the current value of a0
% here are equations 33a-f:
F(1) = -a2*U + 1/2/pi*integral(@(phi0)g1(phi0,x,params),0,2*pi);
F(2) = H - Y + a1*G;
F(3) = -mu*H + (mu + 1)*Y + a2*X - 1/pi*integral(@(phi0)(g1timescos(phi0,x,params)),0,2*pi);
F(4) = -G + X + a1*H + a0;
F(5) = mu*G - (mu + 1)*X + a2*Y - 1/pi*integral(@(phi0)(g1timessin(phi0,x,params)),0,2*pi);

end

