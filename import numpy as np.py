#...................................
#DRAWINNG three samples from the Matern Gaussian process prior
#....................................
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern

# Inputng points
X = np.linspace(-5, 5, 200).reshape(-1,1)

# ----- Matern 3/2 which is the first parameter -----
kernel1 = 1.0**2 * Matern(length_scale=1, nu=1.5)
# first kernel is the Matern kernel with length scale 1 and nu=1.5, which corresponds to the Matern 3/2 kernel  
K1 = kernel1(X)
samples1 = np.random.multivariate_normal(
    mean=np.zeros(len(X)),
    cov=K1,
    size=3
)

# .....Matern 5/2 WHICH IS THE SECOND KERNEL.....
kernel2 = 2.0**2 * Matern(length_scale=2, nu=2.5)
# second kernel is the Matern kernel with length scale 2 and nu=2.5, which corresponds to the Matern 5/2 kernel 
K2 = kernel2(X)
samples2 = np.random.multivariate_normal(
    mean=np.zeros(len(X)),
    cov=K2,
    size=3
)

# Plotting the samples
fig, ax = plt.subplots(1,2, figsize=(16,4))

for s in samples1:
    ax[0].plot(X, s)
#title for the first plot
ax[0].set_title("Matern Gaussian process Prior (ν=3/2)")

for s in samples2:
    ax[1].plot(X, s)
#title for the second plot
ax[1].set_title("Matern Gaussian process Prior (ν=5/2)")

plt.show()