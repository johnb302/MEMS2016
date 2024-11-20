function [dA0] = odefun_A0(t,A0,params)
% keep the numerical solution ic's around to seed solutions after the first
persistent nsolve_ics

% if it is the first iteration in the integration of A0, use these ICs for
% fsolve
if isempty(nsolve_ics)
   nsolve_ics = [100, 100, 100, 100, 100]'; 
end

% assign a0cur so it can be passed through params into the fsolve routine
params.a0cur = A0;
lambda = params.lambda;

% we solve the harmonic balance (fourier coefficient) system (Eq 33) in
% this line
x = fsolve(@(x)hbsystem_case1(x,params),nsolve_ics,params.options);

fsolve_ics = x;

H = x(3);

% in order to compute dA0/d_eta_2, we need H from the fsolve above.
dA0 = (-1/2)*(lambda*A0 - H);
end


