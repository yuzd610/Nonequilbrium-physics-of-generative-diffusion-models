
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

n = 10000

pi1 = 0.5
pi2 = 0.5
mu1 = 1
mu2 = -1
sigma_2 = 1

ds = 0.01
Iterate = 1000
T = ds * Iterate


def sample_mixture_gaussian(pi1, mu1,sigma_2, pi2, mu2,  n_samples):

    assert np.isclose(pi1 + pi2, 1), "The sum of pi1 and pi2 must be 1."


    dist_selection = np.random.choice(['dist1', 'dist2'], size=n_samples, p=[pi1, pi2])


    samples = np.zeros(n_samples)

    # Select and sample each sample
    for i, dist in enumerate(dist_selection):
        if dist == 'dist1':
            samples[i] = np.random.normal(mu1 * np.exp(-T), np.sqrt((1-np.exp(-2*T)+sigma_2*np.exp(-2*T))))
        else:
            samples[i] = np.random.normal(mu2 * np.exp(-T),  np.sqrt((1-np.exp(-2*T)+sigma_2*np.exp(-2*T))))

    return samples

#Calculate score function
def complex_expression(x, t):
    variance_term = 1 - np.exp(-2 * t) + np.exp(-2 * t) * sigma_2

    numerator = np.tanh(mu1*np.exp(-t)*x/(variance_term))*mu1*np.exp(-t)-x


    result = numerator / (variance_term)
    return result











#Sampling the entire trajectory
X = np.zeros((n, Iterate + 1))

X_0 = sample_mixture_gaussian(pi1, mu1, sigma_2, pi2, mu2, n_samples=n)

X[:, 0] = X_0


for i in range(0, Iterate):
    X[:, i + 1] = X[:, i] + (2*complex_expression(X[:, i], (T - i * ds))+X[:, i]) * ds + np.random.normal(0, 1, n) * np.sqrt(ds) * np.sqrt(2)



# Change in entropy of the system
def S(X0, XT):
    S_T = np.log( 0.5 * norm.pdf(XT, loc=mu1 *np.exp(-T), scale=np.sqrt(1-np.exp(-2*T)+sigma_2*np.exp(-2*T))) + 0.5 * norm.pdf(XT, loc=mu2 *np.exp(-T), scale=np.sqrt(1-np.exp(-2*T)+sigma_2*np.exp(-2*T))))
    S_0 = np.log(0.5 * norm.pdf(X0, loc=mu1 , scale=np.sqrt(sigma_2)) + 0.5 * norm.pdf(X0, loc=mu2, scale=np.sqrt(sigma_2)))

    return S_T-S_0

# Entropy flux of the system
def Sm(X):

    results = np.zeros(X.shape[0])


    for i, row in enumerate(X):
        sum_result = 0
        #stratonovich_integral

        for n in range(len(row) - 1):
            x_n = row[n]
            x_n_plus_1 = row[n + 1]
            mid_point = (x_n_plus_1 + x_n) / 2
            diff = (x_n_plus_1 - x_n) / ds



            term = 2*complex_expression(mid_point, (T - n * ds))+ mid_point

            sum_result += diff * term * ds  # Accumulate the results of the calculation

        results[i] = sum_result

    return results


S = S(X[:, -1], X[:, 0])
Sm = Sm(X)

S_tol = S + Sm

mean = np.exp(-S_tol)





label_fontsize = 20  # Font size for axis labels
tick_fontsize = 20   # Font size for axis ticks


# Draw the first histogram
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(S, bins=200, color='skyblue', density=True)
plt.xlabel('$\Delta S$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)

plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.xlim(-5, 5)
plt.ylim(0, 2)

plt.tight_layout()
plt.savefig('1.pdf')
plt.show()

# Draw the second histogram
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(Sm, bins=100, color='lightgreen', density=True)
plt.xlabel('$\Delta S_E$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.tight_layout()
plt.savefig('2.pdf')
plt.show()

# Draw the third histogram
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(S_tol, bins=100, color='salmon', density=True)
plt.xlabel('$\Delta S_{tot}$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.tight_layout()
plt.savefig('3.pdf')
plt.show()

# Draw the fourth histogram
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(mean, bins=300, color='brown', density=True)
plt.xlabel('$e^{-\Delta S_{tot}}$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.xlim(-1, 7) # Set the horizontal axis range
plt.tight_layout()
plt.savefig('4.pdf')
plt.show()







def P(x, t):
    """
    The probability density function at time t is a mixture of two normal distributions.
    The mean of each distribution decays at a rate of exp(-t) over time, and the variance also changes over time.
    """
    return 0.5 * norm.pdf(x, loc=mu1 *np.exp(-t), scale=np.sqrt(1-np.exp(-2*t)+sigma_2*np.exp(-2*t))) + \
           0.5 * norm.pdf(x, loc=mu2 *np.exp(-t), scale=np.sqrt(1-np.exp(-2*t)+sigma_2*np.exp(-2*t)))
x = np.linspace(-8, 8, 1000)
pdf_values =  P(x, 0)










plt.figure(figsize=(5, 4), dpi=200)
plt.hist(X[:, -1 ] , bins=500, color='brown', density=True)
plt.plot(x, pdf_values, color='r', label='PDF')
plt.xlabel('$e^{-\Delta S_{tot}}$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.xlim(-8, 8)  # Set the horizontal axis range
plt.tight_layout()
plt.show()








print(np.var(X[:, -1 ]))



print(np.mean(np.exp(-S_tol)))

print(np.mean(S))











