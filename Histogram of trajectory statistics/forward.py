import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.misc import derivative
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.misc import derivative
from concurrent.futures import ProcessPoolExecutor
n = 10000

pi1 = 0.5
pi2 = 0.5
mu1 = 1
mu2 = -1
sigma_2 = 0.1

ds = 0.001
Iterate = 1000
T = ds * Iterate


def sample_mixture_gaussian(pi1, mu1,sigma_2, pi2, mu2,  n_samples):

    assert np.isclose(pi1 + pi2, 1), "The sum of pi1 and pi2 must be 1."


    dist_selection = np.random.choice(['dist1', 'dist2'], size=n_samples, p=[pi1, pi2])


    samples = np.zeros(n_samples)

    # 对每个采样进行分布选择和采样
    for i, dist in enumerate(dist_selection):
        if dist == 'dist1':
            samples[i] = np.random.normal(mu1 , np.sqrt(sigma_2))
        else:
            samples[i] = np.random.normal(mu2 , np.sqrt(sigma_2))

    return samples











X = np.zeros((n, Iterate + 1))

X_0 = sample_mixture_gaussian(pi1, mu1, sigma_2, pi2, mu2, n_samples=n)

X[:, 0] = X_0


for i in range(0, Iterate):
    X[:, i + 1] = X[:, i] + -X[:, i] * ds + np.random.normal(0, 1, n) * np.sqrt(ds) * np.sqrt(2)




def S(X0, XT):
    S_T = np.log( 0.5 * norm.pdf(XT, loc=mu1 *np.exp(-T), scale=np.sqrt(1-np.exp(-2*T)+sigma_2*np.exp(-2*T))) + 0.5 * norm.pdf(XT, loc=mu2 *np.exp(-T), scale=np.sqrt(1-np.exp(-2*T)+sigma_2*np.exp(-2*T))))
    S_0 = np.log(0.5 * norm.pdf(X0, loc=mu1 , scale=np.sqrt(sigma_2)) + 0.5 * norm.pdf(X0, loc=mu2, scale=np.sqrt(sigma_2)))

    return S_0-S_T








def Sm(matrix):

    # 计算差分 / dt
    diff_dt = np.diff(matrix, axis=1) / ds

    # 计算 (X(n) + X(n + 1)) / 2
    avg_x = (matrix[:, :-1] + matrix[:, 1:]) / 2
    instant_power = -diff_dt*avg_x

    # 计算总和
    result = np.sum(instant_power, axis=1) * ds
    return result

















S = S(X[:, 0],X[:, -1])
Sm = Sm(X)

S_tol = S + Sm

mean = np.exp(-S_tol)





label_fontsize = 20  # 轴标签的字体大小
tick_fontsize = 20   # 轴刻度的字体大小

# 第一个直方图
# 重新绘制直方图，去掉边缘线以提高流畅性

# 绘制第一个直方图
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

# 绘制第二个直方图
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(Sm, bins=100, color='lightgreen', density=True)
plt.xlabel('$\Delta S_E$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.tight_layout()
plt.savefig('2.pdf')
plt.show()

# 绘制第三个直方图
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(S_tol, bins=100, color='salmon', density=True)
plt.xlabel('$\Delta S_{tot}$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.xlim(-3, 2)
plt.tight_layout()

plt.savefig('3.pdf')
plt.show()

# 绘制第四个直方图
plt.figure(figsize=(5, 4), dpi=200)
plt.hist(mean, bins=1500, color='brown', density=True)
plt.xlabel('$e^{-\Delta S_{tot}}$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.xlim(0, 4)

plt.tight_layout()
plt.savefig('4.pdf')
plt.show()



plt.figure(figsize=(5, 4), dpi=200)
plt.hist(X[:, 0 ] , bins=500, color='brown', density=True)
plt.xlabel('$e^{-\Delta S_{tot}}$', fontsize=label_fontsize)
plt.ylabel('Normalized histogram', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.xlim(-5, 5)  # 设置横坐标范围
plt.tight_layout()
plt.show()

def P(x, t):
    """
    时间t下的概率密度函数，为两个正态分布的混合。
    每个分布的均值随时间以exp(-t)的速率衰减，方差也随时间变化。
    """
    return 0.5 * norm.pdf(x, loc=mu1 *np.exp(-t), scale=np.sqrt(1-np.exp(-2*t)+sigma_2*np.exp(-2*t))) + \
           0.5 * norm.pdf(x, loc=mu2 *np.exp(-t), scale=np.sqrt(1-np.exp(-2*t)+sigma_2*np.exp(-2*t)))
x = np.linspace(-5, 5, 1000)
pdf_values =  P(x, 0)

# 绘制直方图
plt.plot(x, pdf_values, color='r', label='PDF')
plt.title('Normal Distribution PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()





print(np.var(X[:, 0]))

print(np.exp(-S_tol))

print(np.mean(np.exp(-S_tol)))

print(np.mean(S))











