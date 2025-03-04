import numpy as np
import matplotlib.pyplot as plt




pi1 = 0.5
pi2 = 0.5
mu1 = -1
mu2 = 1
sigma1 = 1
sigma2 = 1
dt = 0.01
Iterate = 1000

def sample_mixture_gaussian(pi1, mu1, sigma1, pi2, mu2, sigma2, n_samples):

    assert np.isclose(pi1 + pi2, 1), "The sum of pi1 and pi2 must be 1."


    dist_selection = np.random.choice(['dist1', 'dist2'], size=n_samples, p=[pi1, pi2])


    samples = np.zeros(n_samples)


    for i, dist in enumerate(dist_selection):
        if dist == 'dist1':
            samples[i] = np.random.normal(mu1, sigma1)
        else:
            samples[i] = np.random.normal(mu2, sigma2)

    return samples



N = []

Mean =[]






for n in range(0,10001,100):
    N.append(n)
    X = np.zeros((n, Iterate + 1))
    # Sampling initial distribution

    X_0 = sample_mixture_gaussian(pi1, mu1, sigma1, pi2, mu2, sigma2, n_samples=n)

    X[:, 0] = X_0
    # Iterate the complete trajectory

    for i in range(0, Iterate):
        X[:, i + 1] = X[:, i] - X[:, i] * dt + np.random.normal(0, 1, n) * np.sqrt(dt) * np.sqrt(2)


    # Calculate the entropy change of the system


    def S(X0, XT, mu_star, T):
        return np.log(
            (np.exp(2 * X0 * mu_star) + 1) * np.exp(
                -0.5 * X0 ** 2 - X0 * mu_star + 0.5 * XT ** 2 + XT * mu_star * np.exp(
                    -T) - 0.5 * mu_star ** 2 + 0.5 * mu_star ** 2 * np.exp(-2 * T)) /
            (np.exp(2 * XT * mu_star * np.exp(-T)) + 1)
        )


    # Calculate the entropy flux of the system


    def Sm(matrix, dt):

        # Calculate the difference / dt
        diff_dt = np.diff(matrix, axis=1) / dt

        # Calculate (X(n) + X(n + 1)) / 2
        avg_x = (matrix[:, :-1] + matrix[:, 1:]) / 2
        instant_power = -diff_dt * avg_x

        # Calculate the sum
        result = np.sum(instant_power, axis=1) * dt
        return result


    Sm = Sm(X, dt)

    S = S(X[:, 0], X[:, -1], mu2, dt * Iterate)
    # Calculate entropy production

    S_tol = Sm + S

    mean = np.exp(-S_tol)
    Mean.append(np.mean(np.exp(-S_tol)))


print(N)



print(Mean)







plt.plot(N, Mean, marker='o',color='salmon')
plt.xlabel('number of trajectories',fontsize=25)
plt.ylabel('$\left\langle {{e^{ - \Delta {S_{tot}}}}} \\right\\rangle $',fontsize=25)  # Vertical axis label
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)  
plt.tight_layout()
plt.savefig('forward_number.pdf')
plt.show()

























