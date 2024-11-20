% Root Code Driver for the Solution in The Dynamics of MEMS Colpitts
% Oscillators, focused on a single Colpitts system.
% alternative implementation - NBajaj 04062022 - Pitt

clc
clear all
tic

% dimensional parameters

Cm = 2.201e-15;
Cs = 1.62e-11;
Cg = 2.32e-11;
Lm = 4.496e-2;
Rm = 5.62e1;
Rs = 7.444e3;
Rg = 5.168e4;
Vth = -0.95;
beta = 0.12;
B = 0.8;

% options for the fsolve command, to be passed with the parameters struct
% (called params)
params.options = optimoptions(@fsolve,'FunctionTolerance', ... 
    1e-6,'MaxIterations',400,'Display','off');

% computation of nondimensional parameters
mu = Cg/Cs
a1 = sqrt(Lm*Cm)/Cg/Rg
a2 = sqrt(Lm*Cm)/Cs/Rs
ep = sqrt(Cm/Cg)
lambda = Rm*Cg/sqrt(Lm*Cm)
gamma = beta*abs(Vth)*sqrt(Lm*Cm)/Cs

% nondimensional parameters (assignment into params)
params.mu = mu;
params.a1 = a1;
params.a2 = a2;
params.ep = ep;
params.lambda = lambda;
params.gamma = gamma;

% compute the explicit u0 steady state value, derived from the
% nondimensionalized equations (e.g. 19a-c, with the derivatives set to
% zero)
u0 = (1 + params.a2/2/params.gamma) ... 
    - sqrt((1 + params.a2/2/params.gamma)^2 - 1);

v0 = 0;

params.v0 = v0;
params.u0 = u0;

% initial condition on A0
params.a0ic = 0;
params.a0cur = 0;
tspan = [0, 200];

% we focus here on the integration of A0 with time.
[t,a0t] = ode45(@(t,A0)odefun_A0(t,A0,params),tspan,params.a0ic);
toc

%profile off
%profile report
plot(t/ep^2,abs(a0t))
