
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


pi1 = 0.5
pi2 = 0.5
mu1 = 1
mu2 = -1
sigma1 = 1
sigma2 = 1
ds = 0.01
Iterate = 1000
T = ds * Iterate


def sample_mixture_gaussian(pi1, mu1, sigma1, pi2, mu2, sigma2, n_samples):

    assert np.isclose(pi1 + pi2, 1), "The sum of pi1 and pi2 must be 1."


    dist_selection = np.random.choice(['dist1', 'dist2'], size=n_samples, p=[pi1, pi2])


    samples = np.zeros(n_samples)


    for i, dist in enumerate(dist_selection):
        if dist == 'dist1':
            samples[i] = np.random.normal(mu1 * np.exp(-T), sigma1)
        else:
            samples[i] = np.random.normal(mu2 * np.exp(-T), sigma2)

    return samples



N = []

Mean =[]






for n in range(0,10001,100):
    N.append(n)
    X = np.zeros((n, Iterate + 1))

    X_0 = sample_mixture_gaussian(pi1, mu1, sigma1, pi2, mu2, sigma2, n_samples=n)

    X[:, 0] = X_0

    for i in range(0, Iterate):
        X[:, i + 1] = X[:, i] + (2 * np.tanh(mu1 * np.exp(-(T - i * ds)) * X[:, i]) * mu1 * np.exp(-(T - i * ds)) - X[:,
                                                                                                                    i]) * ds + np.random.normal(
            0, 1, n) * np.sqrt(ds) * np.sqrt(2)


    def S(X0, XT, mu_star, T):
        return -np.log(
            (np.exp(2 * X0 * mu_star) + 1) * np.exp(
                -0.5 * X0 ** 2 - X0 * mu_star + 0.5 * XT ** 2 + XT * mu_star * np.exp(
                    -T) - 0.5 * mu_star ** 2 + 0.5 * mu_star ** 2 * np.exp(-2 * T)) /
            (np.exp(2 * XT * mu_star * np.exp(-T)) + 1)
        )


    def Sm(X, ds, mu_star, T):

        results = np.zeros(X.shape[0])

        for i, row in enumerate(X):
            sum_result = 0

            for n in range(len(row) - 1):
                x_n = row[n]  # 当前元素
                x_n_plus_1 = row[n + 1]  # 下一个元素
                mid_point = (x_n_plus_1 + x_n) / 2
                diff = (x_n_plus_1 - x_n) / ds
                exp_term = np.exp(- (T - n * ds))

                term = (2 * np.tanh(mu_star * exp_term * mid_point) * mu_star * exp_term) - mid_point

                sum_result += diff * term * ds  # 累加计算的结果

            results[i] = sum_result

        return results


    S = S(X[:, -1], X[:, 0], mu2, ds * Iterate)
    Sm = Sm(X, ds, mu1, T)

    S_tol = S + Sm

    mean = np.exp(-S_tol)
    Mean.append(np.mean(np.exp(-S_tol)))
    print(n)


print(N)



print(Mean)







plt.plot(N, Mean, marker='o',color='salmon')  # 使用圆圈标记每个数据点
plt.xlabel('number of trajectories',fontsize=25)  # 横轴标签
plt.ylabel('$\left\langle {{e^{ - \Delta {S_{tot}}}}} \\right\\rangle $',fontsize=25)  # 纵轴标签
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)  # 显示网格
plt.tight_layout()
plt.savefig('backward_number.pdf')
plt.show()


