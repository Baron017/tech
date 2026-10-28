#............Question 2d..........
import numpy as np
import matplotlib.pyplot as plt

# Input grid
X = np.linspace(-4, 4, 200)

# Parameters for the kernels
psi = -0.5
taus = [-1, 0, 1]
lam = 3

# ............Basis-function kernel............

# Basis functions for the finite basis-function GP
B = np.array([
    np.exp(psi * (X - tau)**2)
    for tau in taus
])

# Kernel matrix for the basis-function GP
K_basis = B.T @ B

# Small jitter for stability
K_basis += 1e-6 * np.eye(len(X))

# Draw 3 samples from the finite basis-function GP prior
samples_basis = np.random.multivariate_normal(
    mean=np.zeros(len(X)),
    cov=K_basis,
    size=3
)

#  RBF kernel for the RBF GP

X1 = X[:, None]
X2 = X[None, :]
# RBF kernel matrix
Ke_rbf = np.exp(-lam * (X1 - X2)**2)

Ke_rbf += 1e-6 * np.eye(len(X))

samples_rbf = np.random.multivariate_normal(
    mean=np.zeros(len(X)),
    cov=Ke_rbf,
    size=3
)

# ..... Making the Plot......

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Basis-function Gaussian process Prior
for s in samples_basis:
    ax[0].plot(X, s)
# title for the first plot
ax[0].set_title("Finite Basis Gaussian Process Prior")

# RBF Gaussian process Prior
for s in samples_rbf:
    ax[1].plot(X, s)
# title for the second plot
ax[1].set_title("RBF Gaussian Process Prior")
# Adjust layout and show the plot
plt.tight_layout()
plt.show()


