import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def euler_method(f, x0, y0, x_end, h, n_iteration):
    x_values = [x0]
    y_values = [y0]

    for i in range(n_iteration):
        x_current = x_values[-1]
        y_current = y_values[-1]

        slope = f(x_current, y_current)
        
        y_next = y_current + slope * h
        x_next = x_current + h
        
        if x_next > x_end:
            x_next = x_end
            h_last = x_end - x_current
            y_next = y_current + f(x_current, y_current) * h_last
            x_values.append(x_next)
            y_values.append(y_next)
            break
        x_values.append(x_next)
        y_values.append(y_next)

    return x_values, y_values


def derivative(x, y):
    return -0.5 * y  

x0 = 0.0
y0 = 5.0
x_end = 10.0
h = 0.01  
n_iteration = 1000

x_vals, y_vals = euler_method(derivative, x0, y0, x_end, h, n_iteration)

span = np.linspace(0, x_end, int(x_end/h) + 1)

sol = solve_ivp(
    derivative,
    [x0, x_end],
    [y0],
    method='RK45',
    t_eval=span,  
    vectorized=True
)

x_rk45 = sol.t
y_rk45 = sol.y[0]

fig, axs = plt.subplots(1,2, figsize=(16, 8))
ax1 = axs[0]
ax2 = axs[1]
ax1.plot(x_vals, y_vals, 'ro', label='Euler method')
ax1.plot(x_rk45, y_rk45, 'b-', label='RK45 (solve_ivp)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(r"Solution of $\frac{dy}{dx} = -0.5y$ with $y(0) = 5.0$")
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 5)

min_y = min(np.min(y_vals[:]), np.min(y_rk45[:]))
max_y = max(np.max(y_vals[:]), np.max(y_rk45[:]))
ax2.plot([min_y, max_y], [min_y, max_y], 'k--', label='Diagonal line')
ax2.plot(y_vals, y_rk45, 'b.', label='Fitting line')
ax2.set_xlabel('Euler method')
ax2.set_ylabel('RK45')
ax2.set_title("Parity plot")
ax2.set_xlim(min_y, max_y)
ax2.set_ylim(min_y, max_y)
ax2.grid(True)
plt.show()