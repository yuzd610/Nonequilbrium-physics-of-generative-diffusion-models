import numpy as np
import matplotlib.pyplot as plt

def p(x, t, u, sigma):
    term1 = (1 - np.exp(-2 * t)) + sigma**2 * np.exp(-2 * t)
    return 0.5 * (1 / np.sqrt(2 * np.pi * term1)) * np.exp(-0.5 * ((x - u * np.exp(-t))**2) / term1) + \
           0.5 * (1 / np.sqrt(2 * np.pi * term1)) * np.exp(-0.5 * ((x + u * np.exp(-t))**2) / term1)

def U(x, t, u, sigma):
    return -0.5 * x**2 - 2 * np.log(p(x, t, u, sigma))

def t_s(mu, sigma):
    return 0.5 * np.log(mu**2 + np.sqrt(sigma**4 - 2*sigma**2 + mu**4 + 1))

# Define the range of mu and sigma
mu_values = np.linspace(0, 1, 400)
sigma_values = np.linspace(0, 2, 400)
mu, sigma = np.meshgrid(mu_values, sigma_values)

# Calculate t_s and U
ts_values = t_s(mu, sigma)
U1 = U(sigma*20 + mu, 0, mu, sigma)
U2 = U(sigma*19 + mu, 0, mu, sigma)
U = U1 - U2

# Define three conditions
mask1 = (U < 0) & (ts_values > 0)  # U < 0, t_s > 0
mask2 = (U > 0) & (ts_values < 0)  # U < 0, t_s < 0
mask3 = (U > 0) & (ts_values > 0)  # U > 0, t_s > 0


levels1 = np.where(mask1, 1, np.nan)  # Set the mask1 area to 1 and other areas to nan
levels2 = np.where(mask2, 1, np.nan)  # The mask2 area is set to 1, and the other areas are nan
levels3 = np.where(mask3, 1, np.nan)  # Set the mask3 area to 1 and other areas to nan


plt.contour(mu, sigma, ts_values, levels=[0], colors='black', linewidths=3)
plt.contour(mu, sigma, U, levels=[0], colors='black', linewidths=3)
plt.contourf(mu, sigma, levels1, colors=['red'], alpha=0.4)
plt.contourf(mu, sigma, levels2, colors=['green'], alpha=0.4)
plt.contourf(mu, sigma, levels3, colors=['saddlebrown'], alpha=0.4)


# Add figure title and labels
symmetry_breaking = (0.1, 1.7)  # Symmetry breaking points
asymmetric_breaking = (0.05, 1)  # Points where asymmetry is broken
qasymmetric_breaking = (0.3, 0.15)  # Points where qasymmetric is broken

plt.text(symmetry_breaking[0] + 0.02, symmetry_breaking[1], 'Unstable symmetry breaking', fontsize=15, color='blue')
plt.text(asymmetric_breaking[0] + 0.02, asymmetric_breaking[1], 'No symmetry breaking', fontsize=15, color='red')
plt.text(qasymmetric_breaking[0] + 0.02, qasymmetric_breaking[1], ' Symmetry breaking', fontsize=15, color='green')


plt.xlabel("$\mu$", fontsize=25)
plt.ylabel("$\sigma$", fontsize=25)

# Set the axis scale font size
plt.tick_params(axis='both', which='major', labelsize=20)


plt.tight_layout()
plt.savefig("phase_diagram.pdf", format='pdf')
plt.show()


