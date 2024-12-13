import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def g1(phi0, x, params):
    U, G, H, X, Y = x
    V = 0  # Constant

    ep = params['ep']
    gamma = params['gamma']
    u0 = params['u0']

    v1mu1 = (V - U) + (G - X) * np.cos(phi0) + (H - Y) * np.sin(phi0)
    cond = (ep * v1mu1 >= u0 - 1)

    g1out = np.where(cond, gamma * v1mu1 * (2 * (1 - u0) + ep * v1mu1), -gamma / ep * (1 - u0) ** 2)
    return g1out


def g1Cos(phi0, x, params):
    return g1(phi0, x, params) * np.cos(phi0)


def g1Sin(phi0, x, params):
    return g1(phi0, x, params) * np.sin(phi0)


def hbsystem_case1(x, params):
    U, G, H, X, Y = x
    mu = params['mu']
    a1 = params['a1']
    a2 = params['a2']
    a0 = params['a0cur']

    F = np.zeros(5)
    F[0] = -a2 * U + (1 / (2 * np.pi)) * quad(lambda phi0: g1(phi0,x,params),0,2*np.pi)[0]
    F[1] = H - Y + a1 * G
    F[2] = -mu * H + (mu + 1) * Y + a2 * X - (1 / np.pi) * quad(lambda phi0: g1Cos(phi0,x,params),0,2*np.pi)[0]
    F[3] = -G + X + a1 * H + a0
    F[4] = mu * G - (mu + 1) * X + a2 * Y - (1 / np.pi) * quad(lambda phi0: g1Sin(phi0,x,params),0,2*np.pi)[0]
    return F


def odefun(t, A0, params):
    if params['nsolve_ics'] is None:
        params['nsolve_ics'] = np.array([100, 100, 100, 100, 100])

    params['a0cur'] = A0
    lambda_ = params['lambda']

    x = fsolve(hbsystem_case1, params['nsolve_ics'], args=(params,))
    params['nsolve_ics'] = x
    H = x[2]

    return (-1 / 2) * (lambda_ * A0 - H)


# Define the nonlinear drain current
def g(v, u):
    delta = v - u + 1
    return gamma * delta**2 if delta >= 0 else 0


# Define the first-order ODEs
def ode9a(t, y):
    f1, f2, v, u = y  # State variables
    # Dynamics of v and u
    u_prime = (g(v, u) - a2 * u) / (1 + mu)
    v_prime = u_prime - a1 * v - ep * f1

    # Dynamics of f
    f1_prime = f2
    f2_prime = -ep**2 * lambda_ * f2 - f1 + ep * v_prime

    return [f1_prime, f2_prime, v_prime, u_prime]


# Solution driver function
if __name__ == "__main__":
    # Dimensional parameters
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

    # Compute nondimensional parameters
    mu = Cg / Cs
    a1 = np.sqrt(Lm * Cm) / (Cg * Rg)
    a2 = np.sqrt(Lm * Cm) / (Cs * Rs)
    ep = np.sqrt(Cm / Cg)
    lambda_ = Rm * Cg / np.sqrt(Lm * Cm)
    gamma = beta * abs(Vth) * np.sqrt(Lm * Cm) / Cs

    # Compute steady-state value u0
    u0 = (1 + a2 / (2 * gamma)) - np.sqrt((1 + a2 / (2 * gamma)) ** 2 - 1)
    v0 = 0

    params = {
        'mu': mu, 'a1': a1, 'a2': a2, 'ep': ep, 'lambda': lambda_, 'gamma': gamma, 'v0': v0, 'u0': u0,
        'a0cur': 0, 'nsolve_ics': None
    }

    t_span = (0, 200)
    t_eval = np.linspace(*t_span, 50000)

    # Integrate the ODE
    results = solve_ivp(odefun, t_span, [0], t_eval=t_eval, args=(params,))
    solution = solve_ivp(ode9a, t_span, [-100, 0, 100, 100], t_eval=t_eval, method='RK45')

    signal = solution.y[0]
    upperEnvelope = np.zeros(len(signal))
    time = [0, ]
    amp = [signal[0], ]
    for k in range(1, len(signal) - 1):
        if (np.sign(signal[k] - signal[k - 1]) == 1) and (np.sign(signal[k] - signal[k + 1]) == 1):
            time.append(k)
            amp.append(signal[k])

    time.append(len(signal) - 1)
    amp.append(signal[-1])
    upper = interp1d(time, amp, kind='cubic', bounds_error=False, fill_value=0.0)

    for k in range(0, len(signal)):
        upperEnvelope[k] = upper(k)

    t = results.t
    # Plot the results
    plt.plot(t[:25000] / ep ** 2, np.abs(results.y[0])[:25000], color='r', linestyle='--', label='$A_0$')
    plt.plot(t[:25000] / ep ** 2, upperEnvelope[:25000], color='b', label='||f||')
    plt.xlabel('t/ep^2')
    plt.ylabel('|A0|')
    plt.title('Dynamics of A0 over Non-Dimensional Time')
    plt.legend()
    plt.show()
