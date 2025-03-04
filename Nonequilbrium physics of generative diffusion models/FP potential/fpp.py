import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm


mu = 1.0
sigma2 = 0.1
t = 1
num = 800
q = []
v = []
dv = []
ddv =[]

def sample_joint(mu, sigma2, size):
    X0_prime_components = np.random.choice([1, -1], size=size)
    X0_prime_samples = np.random.normal(loc=X0_prime_components * mu, scale=np.sqrt(sigma2), size=size)
    Xt_samples = np.random.normal(loc=X0_prime_samples * np.exp(-t), scale=np.sqrt(1 - np.exp(-2 * t)), size=size)
    return X0_prime_samples, Xt_samples
#2D Monte Carlo sampling
X0_prime_samples, Xt_samples = sample_joint(mu, sigma2, size=100000)

def H(X0, Xt):
    cond_exp_term = (Xt - X0 * np.exp(-t)) ** 2 / (2 * (1 - np.exp(-2 * t)))
    marginal_dist_1 = np.exp(-(X0 - mu) ** 2 / (2 * sigma2))
    marginal_dist_2 = np.exp(-(X0 + mu) ** 2 / (2 * sigma2))
    marginal_distribution_term = -np.log((1 / (2 * np.sqrt(2 * np.pi * sigma2))) * (marginal_dist_1 + marginal_dist_2))
    H_value = cond_exp_term + marginal_distribution_term
    return H_value

#Partition function calculation
def Partition(X0_prime, Xt_samples,Q):
    Partition_function = (1 / (2 * np.sqrt(Q))) * (np.exp(-H(X0_prime + np.sqrt(Q), Xt_samples)) + np.exp(
        -H(X0_prime - np.sqrt(Q), Xt_samples)))
    free = -np.log(Partition_function)
    return free



def first_derivative(X0_prime, Xt_samples, Q, h=1e-5):
    f_plus = Partition(X0_prime, Xt_samples, Q + h)
    f_minus = Partition(X0_prime, Xt_samples, Q - h)
    derivative = (f_plus - f_minus) / (2 * h)
    return derivative




def second_derivative(Q, X0_prime, Xt_samples, h=1e-5):
    f_plus = Partition(X0_prime, Xt_samples, Q + h)
    f_minus = Partition(X0_prime, Xt_samples, Q - h)
    f = Partition(X0_prime, Xt_samples, Q)
    return (f_plus - 2 * f + f_minus) / h**2








def V(Q):
    Q = Q*0.01+0.075
    zero = Partition(X0_prime_samples, Xt_samples,Q)
    # Monte Carlo average

    FP = np.mean(zero)

    first = first_derivative(X0_prime_samples, Xt_samples, Q)
    # Monte Carlo average

    dfp= np.mean(first)




    second = second_derivative(Q, X0_prime_samples, Xt_samples)
    # Monte Carlo average
    ddfp = np.mean(second)




    return Q,FP ,dfp,ddfp
#parallel computing
def integrand():
    results = Parallel(n_jobs=-1)(delayed(V)(i) for i in tqdm(range(num)))
    q1, FP1 ,dfp1,ddfp1= zip(*results)

    # 对返回的结果进行处理，例如：
    for q2, FP2 ,dfp2,ddfp2 in zip(q1, FP1,dfp1,ddfp1):
        q.append(q2)
        v.append(FP2)
        dv.append(dfp2)
        ddv.append(ddfp2)

if __name__ == '__main__':
    integrand()

plt.figure(figsize=(7, 5))
plt.plot(q, v,linewidth=2)
plt.title(f't = {t}, $\sigma ^2$ = {sigma2}',fontsize=30)
plt.xlabel('q',fontsize=30)
plt.ylabel('$V(q)$',fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('1.pdf')
plt.show()



# Plot the first-order derivative of v with respect to q
plt.figure(figsize=(7, 5))
plt.plot(q, dv,linewidth=2)
plt.title(f't = {t}, $\sigma ^2$ = {sigma2}',fontsize=30)
plt.xlabel('q',fontsize=30)
plt.ylabel('$\\frac{{dV(q)}}{{dq}}$',fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('2.pdf')
plt.show()




# Plot the second-order derivative of v with respect to q
plt.figure(figsize=(7, 5))
plt.plot(q, ddv,linewidth=2)
plt.title(f't = {t}, $\sigma ^2$ = {sigma2}',fontsize=30)
plt.xlabel('q',fontsize=30)
plt.ylabel('$\\frac{{{d^2}V(q)}}{{d{q^2}}}$',fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('3.pdf')
plt.show()



















