import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt

#define parameters
alpha = 4e-7
beta = -3e-2
N0 = 40000
ic = np.linspace(0,625,200) # varying gamma and delta/N0 that reflect the paper
ic = np.delete(ic,[72,90,91,92,117,118]) # removing some points that give me weird behavior

def fixed_points4a(alpha, beta, gamma):
    disc = 1 - (4 * gamma * alpha) / (beta ** 2)
    if disc < 0:
        return []
    else:
        return [(-beta / (2 * alpha)) * (1 + np.sqrt(disc)),
                (-beta / (2 * alpha)) * (1 - np.sqrt(disc))]


def cubic_eq(s, alpha, beta, delta):
    return alpha * s**3 + beta * s**2 + delta


def derivative(s, alpha, beta):
    return 3 * alpha * s**2 + 2 * beta * s


def ds_dt(s, alpha, beta, gamma, delta):
    if s <= 0:
        return np.inf
    beta = beta/N0
    delta = delta*N0
    return alpha * s**2 + beta * s + gamma + delta / (s + 1e-6)

# Generate data for Figure 4a
stable_4a = []
unstable_4a = []
stable_4b = []
unstable_4b = []
for value in ic:
    points_4a = fixed_points4a(alpha, beta, value)
    if len(points_4a) == 2:
        stable_4a.append((value, min(points_4a)))
        unstable_4a.append((value, max(points_4a)))

    delta_rescaled = value * N0
    roots = fsolve(cubic_eq,[1,10,100],args=(alpha,beta,delta_rescaled), xtol=1e-6)

    for root in roots:
        if root > 0:
            df_ds = derivative(root,alpha,beta)
            if df_ds > 0:
                unstable_4b.append((value, root))
            else:
                stable_4b.append((value,root))

# Plot Figure 4a
plt.figure(figsize=(12, 6))
if stable_4a:
    gamma_stable, s_stable = zip(*stable_4a)
    plt.plot(gamma_stable, s_stable, 'k-', label='Stable Fixed Point')
if unstable_4a:
    gamma_unstable, s_unstable = zip(*unstable_4a)
    plt.plot(gamma_unstable, s_unstable, 'k--', label='Unstable Fixed Point')
plt.xlim(0,800)
plt.ylim(0,80000)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$s^*$')
plt.title("Figure 4a: Varying Amount of Missions")
plt.legend()
plt.grid()
plt.show()

# Plot Figure 4b
plt.figure(figsize=(12, 6))
if stable_4b:
    delta_stable, s_stable = zip(*stable_4b)
    plt.plot(delta_stable, s_stable, 'k-', label='Stable Fixed Point')
if unstable_4b:
    delta_unstable, s_unstable = zip(*unstable_4b)
    plt.plot(delta_unstable, s_unstable, 'k--', label='Unstable Fixed Point')
plt.xlim(0,800)
plt.ylim(0,80000)
plt.xlabel(r'$\frac{\delta}{N_0}$')
plt.ylabel(r'$s^*$')
plt.title(r'Figure 4b: Constant $\gamma$')
plt.legend()
plt.grid()
plt.show()

# Solving for figures 5a and 5b
beta_N0_0 = np.linspace(-640,0,600)
beta_N0_100 = np.linspace(-740,0,700)
beta_N0_500 = np.linspace(-1140,0,1100)
beta_N0_1600 = np.linspace(-1600,0,1600)
beta_N0_1300 = np.linspace(-1360,0,1300)
beta_N0 = [beta_N0_0, beta_N0_100, beta_N0_500]
beta_N0_5a = [beta_N0_0, beta_N0_100, beta_N0_500, beta_N0_1600]
beta_N0_5b = [beta_N0_0, beta_N0_100, beta_N0_500, beta_N0_1300]
sk = 1/alpha
s0 = 40000

temp = [0,100,500]
values = [0, 100, 500, 1000]

i = 0
time_5a, time_5b = [], []
for value in temp:
    times1 = []
    times2 = []
    for beta in beta_N0[i]:
        time1 = quad(lambda s: 1 / ds_dt(s,alpha,beta,value,0), s0, sk)[0]
        time2 = quad(lambda s: 1 / ds_dt(s,alpha,beta,0,value), s0, sk)[0]
        #if time1 < 450:
        times1.append(time1)
        #if time2 < 450:
        times2.append(time2)
    time_5a.append(times1)
    time_5b.append(times2)
    i = i + 1

times1, times2 = [], []
for beta in beta_N0_1600:
    time1 = quad(lambda s: 1 / ds_dt(s, alpha, beta, 1000, 0), s0, sk)[0]
    times1.append(time1)
time_5a.append(times1)

for beta in beta_N0_1300:
    time2 = quad(lambda s: 1 / ds_dt(s,alpha,beta,0,1000), s0, sk)[0]
    times2.append(time2)
time_5b.append(times2)


# Plot Figure 5a
plt.figure(figsize=(12, 6))
for i, gamma in enumerate(values):
    plt.plot(beta_N0_5a[i], time_5a[i], label=f'$\gamma$ = {gamma}')
plt.xlim(-1600,0)
plt.ylim(0,400)
plt.xlabel(r"$\beta / N_0$")
plt.ylabel(r'$t_k$ (yrs)')
plt.title(r'Figure 5a: Time to Congestion vs $\beta / N_0$')
plt.legend()
plt.grid()
plt.show()

# Plot Figure 5b
plt.figure(figsize=(12, 6))
for i, delta in enumerate(values):
    plt.plot(beta_N0_5b[i], time_5b[i], label=f'$\delta / N_0$ = {delta}')
plt.xlim(-1500,0)
plt.ylim(0,400)
plt.xlabel(r'$\beta / N_0$')
plt.ylabel(r'$t_k$ (yrs)')
plt.title(r'Figure 5b: Time to Congestion vs $\beta / N_0$')
plt.legend()
plt.grid()
plt.show()