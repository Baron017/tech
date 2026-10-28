#Question 3b: Gaussian Process Regression on DJIA log-returns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ExpSineSquared

# 1. Define inputs and targets
y = np.log(np.abs(d) + 1e-8)
X = np.arange(1, len(y) + 1).reshape(-1, 1)

# 2. Standardize X and y for numerical stability during GP fitting
X_scaled = (X - X.mean()) / X.std()
y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std

# Subsample for faster training (Optional, but highly recommended for N > 4000)
# We train the hyper-parameters on every 4th point, but will predict on all points
X_train = X_scaled[::4]
y_train = y_scaled[::4]

# 3. Define the Kernels
k1 = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1.0)
k2 = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1.0)
k3 = RBF() * ExpSineSquared(periodicity=0.1) + WhiteKernel(noise_level=1.0)

models = {
    'RBF + White': GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=2),
    'Matern + White': GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=2),
    'RBF * Periodic + White': GaussianProcessRegressor(kernel=k3, n_restarts_optimizer=2)
}

# 4. Fit, Predict and Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, (name, gp) in enumerate(models.items()):
    gp.fit(X_train, y_train)
    y_pred_scaled, sigma_scaled = gp.predict(X_scaled, return_std=True)
    
    # Unscale predictions back to original domain
    y_pred = y_pred_scaled * y_std + y_mean
    sigma = sigma_scaled * y_std
    
    ax = axes[i]
    ax.scatter(X, y, color='black', alpha=0.1, s=2, label='Observed (Noisy Volatility)')
    ax.plot(X, y_pred, color='blue', lw=2, label='Mean Function $\hat{f}(t)$')
    ax.fill_between(X.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                    color='blue', alpha=0.2, label='95% Posterior Band')
    
    ax.set_title(f'GPR Fit: {name}')
    ax.set_xlabel('Time t (days)')
    ax.set_ylabel('log(|d_t| + 10^-8)')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()