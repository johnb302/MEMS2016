import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from params import params

def g1(phi0,x,p):
    U = x[0]
    V = 0
    G = x[1]
    H = x[2]
    X = x[3]
    Y = x[4]

    ep = p.ep
    gamma = p.gamma
    u0 = p.u0

    # eq 34
    v1mu1 = (V - U) + (G - X)*np.cos(phi0) + (H - Y)*np.sin(phi0)
    # eq 28
    cond = (ep*v1mu1 >= u0 - 1)

    return np.where(cond,gamma*v1mu1*(2*(1-u0)+ep*v1mu1),-1*np.ones(np.size(phi0))*gamma/ep*(1-u0)**2)

def g1Cos(phi0,x,p):
    return g1(phi0,x,p)*np.cos(phi0)

def g1Sin(phi0,x,p):
    return g1(phi0,x,p)*np.sin(phi0)

def hbsystem_case1(x,p):
    # solve this if v1-u1 is not sufficiently large
    mu = p.mu
    a1 = p.a1
    a2 = p.a2
    ep = p.ep
    gamma = p.gamma
    a0 = p.a0cur
    u0 = p.u0

    # corresponding U, V, G, H, X, Y
    U = x[0]
    V = 0
    G = x[1]
    H = x[2]
    X = x[3]
    Y = x[4]

    f1 = -a2*U + 1/2/np.pi*quad(g1,0,2*np.pi,args=(x,p))[0]
    f2 = (H - Y + a1*G).item()
    f3 = -mu*H + (mu + 1)*Y + a2*X - 1/np.pi*quad(g1Cos,0,2*np.pi,args=(x,p))[0]
    f4 = (-G + X + a1*H + a0).item()
    f5 = mu*G - (mu + 1)*X + a2*Y - 1/np.pi*quad(g1Sin,0,2*np.pi,args=(x,p))[0]

    return [f1, f2, f3, f4, f5]

def odefun(A0, t, p):
    if not hasattr(odefun, "nsolve_ics"):
        odefun.nsolve_ics = [100, 100, 100, 100, 100]

    p.setA0cur(A0)
    lmbda = p.lmbda
    # solve harmonic balance
    x = fsolve(hbsystem_case1,odefun.nsolve_ics,args=(p,),xtol=1e-6,maxfev=400)
    odefun.nsolve_ics = x
    H = x[2]
    return -0.5*(lmbda*A0 - H)

if __name__ == '__main__':
    # dimensional parameters
    Cm = 2.201e-15
    Cs = 1.62e-11
    Cg = 2.32e-11
    Lm = 4.496e-2
    Rm = 5.62e1
    Rs = 7.444e3
    Rg = 5.168e4
    Vth = -0.95
    beta = 0.12
    B = 0.8

    # computation of nondimensional parameters
    mu = Cg/Cs
    a1 = np.sqrt(Lm*Cm)/Cg/Rg
    a2 = np.sqrt(Lm*Cm)/Cs/Rs
    ep = np.sqrt(Cm/Cg)
    lmbda = Rm*Cg/np.sqrt(Lm*Cm)
    gamma = beta*np.abs(Vth)*np.sqrt(Lm*Cm)/Cs

    # compute u0 steady state value
    u0 = (1 + a2/2/gamma) - np.sqrt((1 + a2/2/gamma)**2 - 1)
    v0 = 0

    P = params(mu,a1,a2,ep,lmbda,gamma,v0,u0,0,0)

    results = solve_ivp(odefun, [0, 200], [P.a0ic], method='RK45', args=(P,))
    plt.plot((results.t /ep**2), np.abs(results.y[0]))
    plt.xlabel("Tau")
    plt.ylabel("A0")
    plt.show()
    