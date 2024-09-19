import numpy as np
import matplotlib.pyplot as plt

# Define the potential energy function U
def U(X, t):
    return 0.5 * X**2 - 2 * np.log(np.cosh(X * np.exp(-t)))

# Generate X values
X = np.linspace(-3, 3, 400)

# Calculate U for different t values
t_values = [1, 0.35, 0.1]
U_values = [U(X, t) for t in t_values]

# Plotting
# Adjusting font sizes for ticks, labels, and titles
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for ax, U_val, t in zip(axs, U_values, t_values):
    ax.plot(X, U_val, linewidth=3)
    ax.set_title(f' t = {t}', fontsize=30)
    ax.set_xlabel('X', fontsize=30)
    ax.set_ylabel('U', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)


plt.tight_layout()
plt.savefig('potential.pdf')
plt.show()
