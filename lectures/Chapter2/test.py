import numpy as np
import matplotlib.pyplot as plt

# Logistic loss and gradient for separable case
def logistic_loss(w):
    return np.log(1 + np.exp(w))

def logistic_grad(w):
    return 1 / (1 + np.exp(-w))  # sigmoid(w)

# Gradient descent settings
alpha = 0.1
w0 = 0.0
num_iters = 30

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
w_grid = np.linspace(-8, 2, 300)
loss_grid = [logistic_loss(wi) for wi in w_grid]

plt.figure(figsize=(8, 5))
plt.plot(w_grid, loss_grid, label="Logistic Loss", linewidth=2)
plt.plot(w_vals, loss_vals, 'ro--', label="GD Iterates")
plt.xlabel("$w$")
plt.ylabel("$L(w)$")
plt.title("Gradient Descent on Separable 1D Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()