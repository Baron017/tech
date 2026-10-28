import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns

# Your newly installed scikit-learn!
from sklearn.linear_model import LinearRegression

print("All libraries loaded successfully! Hello, world!")

random_seed = 8927
np.random.seed(random_seed)

# Load your hospital data
data = pd.read_csv("manhattan.csv")

# Print the first 5 rows to the terminal so you can see it
print("Here is a peek at your data:")
print(data.head())

# Plot the columns that actually exist in your CSV
sns.scatterplot(x="Cases", y="Deaths", data=data)
plt.show()