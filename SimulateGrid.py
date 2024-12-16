import numpy as np
import sympy as sp
from scipy.integrate import odeint
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

# define necessary functions and symbols
t = sp.symbols('t')
x = sp.Function('x')(t)
y = sp.Function('y')(t)
b, h = sp.symbols('b, h')

# define the vector X and its derivatives
X = sp.Matrix([x,y])
X_dot = X.diff(t)
X_ddot = X_dot.diff(t)

# define X_n, assuming unit distance is 0.75 units
X1 = sp.Matrix([[.75],[.75]])
X2 = sp.Matrix([[-.75],[.75]])
X3 = sp.Matrix([[-.75],[-.75]])
X4 = sp.Matrix([[.75],[-.75]])
Xn = [X1, X2, X3, X4]

# define the system
forcingTerms = sp.Matrix([[0],[0]])
for i in Xn:
    numerator = i - X
    denominator = (numerator.norm()**2 + h**2)**(5/2)
    forcingTerms += numerator/denominator
system = sp.Eq(X_ddot + b*X_dot + X, forcingTerms)
#display(system)

# separate X into two systems
system = sp.Eq(system.lhs - b*X_dot - X, -b*X_dot - X + forcingTerms)
x_sys = sp.Eq(system.lhs[0], system.rhs[0])
y_sys = sp.Eq(system.lhs[1], system.rhs[1])

# Constants
b_value = 0.150  # change damping coefficient for other graphs
h = 0.5        # average height of mass
Xn = [(0.75, 0.75), (-0.75, 0.75), (-0.75, -0.75), (0.75, -0.75)]  # Magnet positions

# Define the system behavior
def dxdy_dt(ic, t):
    x, y, vx, vy = ic
    fx, fy = 0, 0
    for mx, my in Xn:
        dx, dy = mx - x, my - y
        denom = (dx ** 2 + dy ** 2 + h ** 2 + 1e-9) ** (5 / 2)
        fx += dx / denom
        fy += dy / denom
    return [vx, vy, -b_value * vx - x + fx, -b_value * vy - y + fy]

# Determine the magnet to which a trajectory converges
def final_magnet(y_end):
    x_end, y_end = y_end[0], y_end[1]
    distances = [np.hypot(x_end - mx, y_end - my) for mx, my in Xn]
    return np.argmin(distances)  # Returns the index of the closest magnet

# Function to simulate a single initial condition and determine the final magnet
def simulate_point(initial_condition, t):
    trajectory = odeint(dxdy_dt, initial_condition, t)
    magnet_index = final_magnet(trajectory[-1])
    return magnet_index

# Function to run simulations in parallel for a 500x500 grid
def parallel_simulations(grid_size=500):
    # Set up grid of initial conditions
    x_vals = np.linspace(-2, 2, grid_size)
    y_vals = np.linspace(-2, 2, grid_size)
    initial_conditions = [[x, y, 0, 0] for x in x_vals for y in y_vals]

    # Set time vector for integration
    t = np.linspace(0, 100, 1000)

    # Run parallel simulations using joblib
    results = Parallel(n_jobs=-1)(
        delayed(simulate_point)(ic, t) for ic in tqdm(initial_conditions, desc="Running Simulations")
    )

    # Convert results to a 2D array for coloring
    return np.array(results).reshape(grid_size, grid_size)

# Color the grid based on magnet index
def create_color_map(convergence_grid):
    color_map = np.zeros((convergence_grid.shape[0], convergence_grid.shape[1], 3))  # RGB color map

    # Apply colors based on convergence
    for i in range(convergence_grid.shape[0]):
        for j in range(convergence_grid.shape[1]):
            if convergence_grid[i][j] == 0:
                color_map[i][j] = [0,0,0]
            elif convergence_grid[i][j] == 1:
                color_map[i][j] = [1,1,1]
            elif convergence_grid[i][j] == 2:
                color_map[i][j] = [1,1,0]
            else:
                color_map[i][j] = [1,0,0]
    return color_map

if __name__ == '__main__':
    # Run parallel simulations on a 500x500 grid
    convergence_grid = parallel_simulations(grid_size=500)

    # Create color map based on convergence results
    color_map = create_color_map(convergence_grid)

    # Plot the basin of attraction
    plt.figure(figsize=(8, 8))
    plt.imshow(color_map, origin='lower', extent=(-2, 2, -2, 2))
    plt.xlabel('Initial x')
    plt.ylabel('Initial y')
    plt.title('Basins of Attraction for the Magnetic Pendulum')
    plt.show()