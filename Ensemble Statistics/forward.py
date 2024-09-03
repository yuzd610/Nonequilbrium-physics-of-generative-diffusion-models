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
from tqdm import tqdm

n = 100000

pi1 = 0.5
pi2 = 0.5
mu1 = 1
mu2 = -1
sigma_2 = 0.4
num = 200
ds = 0.01
Iterate = 300
T = ds*Iterate


def Stot(t):
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
    mean = np.tanh(mu1*np.exp(-t)/var*sample)**2
    mean1 = np.mean(mean)

    pi = scale + var -2 - scale/(var**2) + scale/(var**2)*mean1 + 1/var
    phi = scale +var -1












    return pi, phi


t_valuesS = np.linspace(0, T, num)


S_values_Sol = [Stot(t)[0] for t in tqdm(t_valuesS)]
S_values_phi_b = [Stot(t)[1]for t in tqdm(t_valuesS)]


S_values_Rate = [a - b for a, b in zip(S_values_Sol,S_values_phi_b)]


fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('t', color='black', fontsize=24)  # 增大xlabel字体大小
ax1.set_ylabel('$value$', color='black', fontsize=24)  # 增大ylabel字体大小

ax1.plot(t_valuesS, S_values_phi_b, label='$\phi (t)$', color=color, linewidth=2)
ax1.plot(t_valuesS, S_values_Sol, label='$\pi (t)$', color='tab:orange', linewidth=2)
ax1.plot(t_valuesS, S_values_Rate, label='${\dot S}(t)$', color='tab:olive', linewidth=2)



ax1.tick_params(axis='y', labelcolor='black', labelsize=20)  # 增大y轴刻度字体大小
ax1.tick_params(axis='x', labelcolor='black', labelsize=20)  # 增大x轴刻度字体大小




plt.axhline(y=0, color='tab:brown', linestyle='--', linewidth=2)




fig.tight_layout()
plt.subplots_adjust(top=0.9)



fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, fontsize='x-large')  # 增大图例字体大小

plt.savefig('1.pdf')



plt.show()








