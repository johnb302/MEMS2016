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

    ep = p['ep']
    gamma = p['gamma']
    u0 = p['u0']

    # eq 34
    v1mu1 = (V - U) + (G - X)*np.cos(phi0) + (H - Y)*np.sin(phi0)
    # eq 28
    cond = (ep*v1mu1 >= u0 - 1)

    g1out = np.where(cond, gamma * v1mu1 * (2 * (1 - u0) + ep * v1mu1), -gamma / ep * (1 - u0) ** 2)
    return g1out

def g1Cos(phi0,x,p):
    return g1(phi0,x,p)*np.cos(phi0)

def g1Sin(phi0,x,p):
    return g1(phi0,x,p)*np.sin(phi0)

def hbsystem_case1(x,p):
    # solve this if v1-u1 is not sufficiently large
    mu = p['mu']
    a1 = p['a1']
    a2 = p['a2']
    a0 = p['a0cur']

    # corresponding U, V, G, H, X, Y
    U, G, H, X, Y = x

    F = np.zeros(5)
    F[0] = -a2*U + (1/(2 * np.pi)) * quad(lambda phi0: g1(phi0,x,p),0,2*np.pi)[0]
    F[1] = H - Y + a1*G
    F[2] = -mu*H + (mu + 1)*Y + a2*X - (1/np.pi) * quad(lambda phi0: g1Cos(phi0,x,p),0,2*np.pi)[0]
    F[3] = -G + X + a1*H + a0
    F[4] = mu*G - (mu + 1)*X + a2*Y - (1/np.pi) * quad(lambda phi0: g1Sin(phi0,x,p),0,2*np.pi)[0]
    return F

def odefun(A0, t, p):
    if p['nsolve_ics'] is None:
        p['nsolve_ics'] = np.array([100, 100, 100, 100, 100])

    p['a0cur'] = A0
    print(p['a0cur'])
    lmbda = p['lmbda']
    # solve harmonic balance
    try:
        x = fsolve(hbsystem_case1, p['nsolve_ics'], args=(p,))
        p['nsolve_ics'] = x  # Update for the next iteration
        H = x[2]
    except Exception as e:
        raise ValueError(f"Error in solving hbsystem_case1: {e}")

    return -0.5*(lmbda*A0 - H)

def sol_driver():
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
    a1 = np.sqrt(Lm*Cm)/(Cg * Rg)
    a2 = np.sqrt(Lm*Cm)/(Cs * Rs)
    ep = np.sqrt(Cm/Cg)
    lmbda = Rm*Cg/np.sqrt(Lm*Cm)
    gamma = beta*abs(Vth)*np.sqrt(Lm*Cm)/Cs

    # compute u0 steady state value
    u0 = (1 + a2/(2 * gamma)) - np.sqrt((1 + a2/(2 * gamma))**2 - 1)
    v0 = 0

    params = {
        'mu': mu, 'a1': a1, 'a2': a2, 'ep': ep, 'lmbda': lmbda, 'gamma': gamma, 'v0': v0, 'u0': u0,
        'a0cur': 0, 'nsolve_ics': None
    }

    results = solve_ivp(odefun, [0, 200], [0], method='RK45', args=(params,), dense_output=True)
    plt.plot(results.t / ep ** 2, np.abs(results.y[0]))
    plt.xlabel("t/ep^2")
    plt.ylabel("|A0|")
    plt.show()

if __name__ == "__main__":
    sol_driver()