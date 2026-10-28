# Question 3: Log-returns of Dow Jones Industrial Average (DJIA) in 2016
import pandas as pd
import numpy as np

# Load the dataset and print its shape and the first few rows to understand its structure
# displaying the first few rows of the dataset to understand its structure
df = pd.read_csv('Dow2016dailyAss2.csv')
print("Shape:", df.shape)
print(df.head())
print(df.tail())
# 1. Reverse the order of the DataFrame to have the oldest date first
df = df.iloc[::-1].reset_index(drop=True)

# 2. Extract Adjusted Close prices as a numpy array
p = df['Adj Close'].values

# 3. Calculate log-returns: d_t = log(P_t) - log(P_{t-1})
drg = np.log(p[1:] / p[:-1])

# 4. Calculate sample mean and sample variance (ddof=1 for sample variance)
d_mean = np.mean(drg)
d_var = np.var(drg, ddof=1)

print(f"Sample Mean of log-returns: {d_mean:.6f}")
print(f"Sample Variance of log-returns: {d_var:.6f}")

#question 3b: Plotting the log-returns
# Fitting Gaussian Process Regression (GPR) models to the log-returns data using different kernels and plot the results.    
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ExpSineSquared

# 1. Define inputs and targets for GPR
y = np.log(np.abs(drg) + 1e-8)
X = np.arange(1, len(y) + 1).reshape(-1, 1)

# 2. Standardize X and y for numerical stability during GP fitting 
X_scaled = (X - X.mean()) / X.std()
y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std

# Select every 4th point for training to reduce computational load
X_train = X_scaled[::4]
y_train = y_scaled[::4]

# 3. Define the Kernels for the GPR models
k1 = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=1.0)
k2 = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5) + WhiteKernel(noise_level=1.0)
k3 = RBF() * ExpSineSquared(periodicity=0.1) + WhiteKernel(noise_level=1.0)
# Creating a dictionary of models for easy iteration
models = {
    'RBF + White': GaussianProcessRegressor(kernel=k1, n_restarts_optimizer=2),
    'Matern + White': GaussianProcessRegressor(kernel=k2, n_restarts_optimizer=2),
    'RBF * Periodic + White': GaussianProcessRegressor(kernel=k3, n_restarts_optimizer=2)
}

# 4. Fit, Predict and Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# Loop through each model, fit it to the training data, make predictions, and plot the results
for i, (name, gp) in enumerate(models.items()):
    gp.fit(X_train, y_train)
    y_pred_scaled, sigma_scaled = gp.predict(X_scaled, return_std=True)
    
# Rescale predictions back to original scale
    y_pred = y_pred_scaled * y_std + y_mean
    sigma = sigma_scaled * y_std
# Plotting the observed data, mean function, and 95 per cent posterior band    
    ax = axes[i]
    ax.scatter(X, y, color='red', alpha=0.1, s=2, label='Observed (Noisy Volatility)')
    ax.plot(X, y_pred, color='blue', lw=2, label='Mean Function $\hat{f}(t)$')
    ax.fill_between(X.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                    color='green', alpha=0.2, label='95 per cent Posterior Band')
 # title and labels for the plot   
    ax.set_title(f'GPR Fit: {name}')
    ax.set_xlabel('Time t (days)')
    ax.set_ylabel('log(|d_t| + 10^-8)')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

#.........Question 3c: Plotting the relationship between log-returns and volumes.........
# Extract volumes and align array sizes
volumes = df['Volume'].values
log_V = np.log(volumes[1:]) 
abs_d = np.abs(drg)
# Creating a scatter plot of log-volumes vs absolute log-returns
plt.figure(figsize=(8, 6))
plt.scatter(log_V, abs_d, alpha=0.4, s=10, color='purple')
plt.title("Absolute Log-returns vs Log-volumes")
plt.xlabel("Log Volume, $\log(V_t)$")
plt.ylabel("Absolute Log-return, $|d_t|$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()