
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

from tqdm import tqdm

n = 100000

pi1 = 0.5
pi2 = 0.5
mu1 = 1
mu2 = -1
sigma_2 = 0.4
num = 200

T = 3


def Stot(t):
    # Monte Carlo Sampling
    def sample_mixture_gaussian(t):

        assert np.isclose(pi1 + pi2, 1), "The sum of pi1 and pi2 must be 1."

        dist_selection = np.random.choice(['dist1', 'dist2'], size=n, p=[pi1, pi2])

        samples = np.zeros(n)

        for i, dist in enumerate(dist_selection):
            if dist == 'dist1':
                samples[i] = np.random.normal(mu1 * np.exp(-t),
                                              np.sqrt((1 - np.exp(-2 * t) + sigma_2 * np.exp(-2 * t))))
            else:
                samples[i] = np.random.normal(mu2 * np.exp(-t),
                                              np.sqrt((1 - np.exp(-2 * t) + sigma_2 * np.exp(-2 * t))))

        return samples

    def P(x, t):

        return 0.5 * norm.pdf(x, loc=mu1 * np.exp(-t), scale=np.sqrt(1 - np.exp(-2 * t) + sigma_2 * np.exp(-2 * t))) + \
            0.5 * norm.pdf(x, loc=mu2 * np.exp(-t), scale=np.sqrt(1 - np.exp(-2 * t) + sigma_2 * np.exp(-2 * t)))


    sample = sample_mixture_gaussian(t)
    var = 1 -np.exp(-2 * t)+sigma_2*np.exp(-2 * t)
    scale = mu1 * mu1 * np.exp(-2 * t)
    # Monte Carlo average
    mean = np.tanh(mu1*np.exp(-t)/var*sample)**2
    mean1 = np.mean(mean)

    pi = scale + var -2 - scale/(var**2) + scale/(var**2)*mean1 + 1/var
    phi = scale + var -3-2*scale/(var**2) + 2*scale/(var**2)*mean1 + 2/var












    return pi, phi


t_valuesS = np.linspace(0, T, num)

#Drawing at different times
S_values_Sol = [Stot(t)[0] for t in tqdm(t_valuesS)]
S_values_phi_b = [Stot(t)[1]for t in tqdm(t_valuesS)]



#Plot t_s
def t_s(mu, sigma):
    return 0.5 * np.log(mu**2 + np.sqrt(sigma**4 - 2 * sigma**2 + mu**4 + 1))






S_values_Rate = [a - b for a, b in zip(S_values_Sol,S_values_phi_b)]


ts = t_s(mu1, np.sqrt(sigma_2))
ts =round(ts,8)
min_value = min(S_values_phi_b)
min_index = S_values_phi_b.index(min_value)
t_min = min_index *(T/(num-1))

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('t', color='black', fontsize=24)  # 增大xlabel字体大小
ax1.set_ylabel('$value$', color='black', fontsize=24)  # 增大ylabel字体大小


ax1.plot(t_valuesS, S_values_phi_b, label=r'${\phi^*}(t)$', color=color, linewidth=2)
ax1.plot(t_valuesS, S_values_Sol, label=r'${\pi^*}(t)$', color='tab:orange', linewidth=2)
ax1.plot(t_valuesS, S_values_Rate, label=r'${{\dot S}^*}(t)$', color='tab:olive', linewidth=2)

ax1.tick_params(axis='y', labelcolor='black', labelsize=20)  # 增大y轴刻度字体大小
ax1.tick_params(axis='x', labelcolor='black', labelsize=20)  # 增大x轴刻度字体大小



plt.axvline(x=ts, color='tab:green', linestyle='--', label=f'${{t_s}}$ = {ts}', linewidth=2)
plt.axhline(y=0, color='tab:brown', linestyle='--', linewidth=2)


plt.axvline(x= t_min, color='purple', linestyle='--', label=f'$t^*=\\arg\\min _t  \, \phi^*(t)$ ', linewidth=2)



fig.tight_layout()  # 调整布局
plt.subplots_adjust(top=0.9)  # 调整顶部空间以避免标题被挡住
# 增加图例字体大小


fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, fontsize='x-large')  # 增大图例字体大小

plt.savefig('figure081.pdf')

plt.show()









