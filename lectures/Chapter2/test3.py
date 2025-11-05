import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x**4 - 4*x**2
def df(x):
    return 4*x**3 - 8*x

start_x = 3; learning_rate = 0.01; n_iterations = 50

trajectory_x = [start_x]; trajectory_y = [f(start_x)]

x = start_x
for i in range(n_iterations):
    gradient = df(x)
    # Gradient descent step!
    x = x - learning_rate * gradient
    
    # Store the new position to plot later
    trajectory_x.append(x); trajectory_y.append(f(x))

    if i % 5 == 0:
        print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}, gradient = {gradient:.4f}")

# Plot the function and trajectory
x_range = np.linspace(-1.5, 3.5, 400); y_range = f(x_range)
plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(12, 7))
plt.plot(x_range, y_range, label='f(x) = $x^4 - 4x^2$', color='royalblue', linewidth=2)
plt.plot(trajectory_x, trajectory_y, 'o-', color='tomato', label='Gradient Descent Trajectory', markersize=5)

# Start and end points
plt.scatter(trajectory_x[0], trajectory_y[0], color='green', s=100, zorder=5, label='Start')
plt.scatter(trajectory_x[-1], trajectory_y[-1], color='darkred', s=100, zorder=5, label='End (Minimum)')

# Title & labels
plt.xlabel('x', fontsize=12); plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=10); plt.grid(True)

# Display the final plot
plt.show()