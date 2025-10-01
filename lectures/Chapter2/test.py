import numpy as np
import matplotlib.pyplot as plt

# Logistic loss and its gradient
def logistic_loss(w):
    return 0.5 * np.log(1 + np.exp(-w)) + 0.5 * np.log(1 + np.exp(2 * w))

def logistic_grad(w):
    term1 = -0.5 / (1 + np.exp(w))
    term2 = 0.5 * 2 / (1 + np.exp(-2 * w))
    return term1 + term2

# Gradient descent parameters
alpha = 0.1      # step size
w0 = 0.0         # initial point
num_iters = 20   # number of iterations

# Store iterates
w_vals = [w0]
loss_vals = [logistic_loss(w0)]

# Run gradient descent
w = w0
for _ in range(num_iters):
    grad = logistic_grad(w)
    w = w - alpha * grad
    w_vals.append(w)
    loss_vals.append(logistic_loss(w))

# Plotting
w_grid = np.linspace(-3, 3, 300)
loss_grid = [logistic_loss(wi) for wi in w_grid]

plt.figure(figsize=(8, 5))
plt.plot(w_grid, loss_grid, label="Logistic Loss", linewidth=2)
plt.plot(w_vals, loss_vals, 'ro--', label="GD Iterates")
plt.xlabel("$w$")
plt.ylabel("$L(w)$")
plt.title("Gradient Descent on Logistic Regression (1D)")
plt.legend()
plt.grid(True)
plt.show()
