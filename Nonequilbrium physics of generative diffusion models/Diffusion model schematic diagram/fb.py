
import numpy as np

import matplotlib.pyplot as plt

each_nums = 1000
A_mean = -4
A_std = 1
B_mean = 4
B_std = 1
Iterate = 300
dt = 0.01
T = Iterate*dt
diffused_steps = Iterate+1
ds = dt

def get_gaussian_mixture_data_numpy(each_nums, A_mean, A_std, B_mean, B_std):
    # Generate samples for distribution A
    A_samples = np.random.normal(loc=[A_mean, A_mean], scale=[A_std, A_std], size=(each_nums, 2))

    # Generate samples for distribution B
    B_samples = np.random.normal(loc=[B_mean, B_mean], scale=[B_std, B_std], size=(each_nums, 2))

    # Concatenate samples from both distributions
    gaussian_mixture_data = np.vstack((A_samples, B_samples))

    return gaussian_mixture_data




def get_gaussian_mixture(each_nums, A_mean, A_std, B_mean, B_std):
    # Generate samples for distribution A
    A_samples = np.random.normal(loc=[A_mean* np.exp(-T), A_mean* np.exp(-T)], scale=[A_std, A_std], size=(each_nums, 2))

    # Generate samples for distribution B
    B_samples = np.random.normal(loc=[B_mean* np.exp(-T), B_mean* np.exp(-T)], scale=[B_std, B_std], size=(each_nums, 2))

    # Concatenate samples from both distributions
    gaussian_mixture_data = np.vstack((A_samples, B_samples))

    return gaussian_mixture_data
def gaussian_matrix(a, b):
    # Generate a matrix with standard Gaussian (normal) distribution elements
    matrix = np.random.randn(a, b)
    return matrix
# Example usage

fx_0 = get_gaussian_mixture_data_numpy(each_nums, A_mean, A_std, B_mean, B_std)

fx = np.zeros((Iterate+1, 2*each_nums, 2))

fx[0, :, :] = fx_0

for i in range(0, Iterate):
    fx[i+1 , :, :] = fx[i , :, :] - fx[i , :, :] * dt + gaussian_matrix(2*each_nums, 2) * np.sqrt(dt) * np.sqrt(2)


bx_0 = get_gaussian_mixture(each_nums, A_mean, A_std, B_mean, B_std)

bx = np.zeros((Iterate+1, 2*each_nums, 2))

bx[0, :, :] = bx_0

for i in range(0, Iterate):
    for j in range(0, 2*each_nums):
        bx[i + 1, j, :] = bx[i, j, :] + (
                    2 * np.tanh(np.sum(A_mean * np.exp(-(T - i * ds)) * bx[i, j, :])) * A_mean * np.exp(-(T - i * ds)) - bx[i, j,
                                                                                                                :]) * ds + gaussian_matrix(
            1, 2) * np.sqrt(ds) * np.sqrt(2)


fig = plt.figure(dpi=300)
ax = plt.axes(projection='3d')



original_data_x = fx_0[:, 0]
original_data_y = np.zeros([each_nums * 2])
original_data_z = fx_0[:, 1]

diffused_data_x = fx[-1,:, 0]
diffused_data_y = np.zeros([each_nums * 2]) + 1
diffused_data_z = fx[-1,:, 1]

resampled_data_x = bx_0[:, 0]
resampled_data_y = np.zeros([each_nums * 2]) + 1.2
resampled_data_z = bx_0[:, 1]

reversed_data_x = bx[-1,:, 0]
reversed_data_y = np.zeros([each_nums * 2]) + 2.2
reversed_data_z = bx[-1,:, 1]

ax.scatter(original_data_x, original_data_y, original_data_z, color='deepskyblue', s=0.5, alpha=0.6, zorder=9)
ax.scatter(diffused_data_x, diffused_data_y, diffused_data_z, color='dodgerblue', s=0.5, alpha=0.6, zorder=2)
ax.scatter(resampled_data_x, resampled_data_y, resampled_data_z, color='mediumseagreen', s=0.5, alpha=0.6, zorder=2)
ax.scatter(reversed_data_x, reversed_data_y, reversed_data_z, color='limegreen', s=0.5, alpha=0.6, zorder=2)

diff_id_1 = 123
diff_id_2 = 234
diff_id_3 = 345
diff_id_4 = 456
diff_id_5 = 567
diff_id_6 = 678

diff_y = np.linspace(0, 1, diffused_steps)

diff_1_x = fx[:, diff_id_1, 0]
diff_1_z = fx[:, diff_id_1, 1]

diff_2_x = fx[:, diff_id_2, 0]
diff_2_z = fx[:, diff_id_2, 1]

diff_3_x = fx[:, diff_id_3, 0]
diff_3_z = fx[:, diff_id_3, 1]

diff_4_x = fx[:, diff_id_4, 0]
diff_4_z = fx[:, diff_id_4, 1]

diff_5_x = fx[:, diff_id_5, 0]
diff_5_z = fx[:, diff_id_5, 1]

diff_6_x = fx[:, diff_id_6, 0]
deno_6_z = fx[:, diff_id_6, 1]

deno_id_1 = 123
deno_id_2 = 234
deno_id_3 = 345
deno_id_4 = 456
deno_id_5 = 567
deno_id_6 = 678

deno_y = np.linspace(1.2, 2.2, diffused_steps)

deno_1_x = bx[:, deno_id_1, 0]
deno_1_z = bx[:, deno_id_1, 1]

deno_2_x = bx[:, deno_id_2, 0]
deno_2_z = bx[:, deno_id_2, 1]

deno_3_x = bx[:, deno_id_3, 0]
deno_3_z = bx[:, deno_id_3, 1]

deno_4_x = bx[:, deno_id_4, 0]
deno_4_z = bx[:, deno_id_4, 1]

deno_5_x = bx[:, deno_id_5, 0]
deno_5_z = bx[:, deno_id_5, 1]

deno_6_x = bx[:, deno_id_6, 0]
deno_6_z = bx[:, deno_id_6, 1]

diff_color = 'lightskyblue'
ax.plot(diff_1_x, diff_y, diff_1_z, linewidth=0.5, zorder=7, color=diff_color)
ax.plot(diff_2_x, diff_y, diff_2_z, linewidth=0.5, zorder=7, color=diff_color)
ax.plot(diff_3_x, diff_y, diff_3_z, linewidth=0.5, zorder=7, color=diff_color)
ax.plot(diff_4_x, diff_y, diff_4_z, linewidth=0.5, zorder=7, color=diff_color)
ax.plot(diff_5_x, diff_y, diff_5_z, linewidth=0.5, zorder=7, color=diff_color)
ax.plot(diff_6_x, diff_y, deno_6_z, linewidth=0.5, zorder=7, color=diff_color)

deno_color = 'mediumseagreen'
ax.plot(deno_1_x, deno_y, deno_1_z, linewidth=0.5, alpha=0.7, zorder=3, color=deno_color)
ax.plot(deno_2_x, deno_y, deno_2_z, linewidth=0.5, alpha=0.7, zorder=3, color=deno_color)
ax.plot(deno_3_x, deno_y, deno_3_z, linewidth=0.5, alpha=0.7, zorder=3, color=deno_color)
ax.plot(deno_4_x, deno_y, deno_4_z, linewidth=0.5, alpha=0.7, zorder=3, color=deno_color)
ax.plot(deno_5_x, deno_y, deno_5_z, linewidth=0.5, alpha=0.7, zorder=3, color=deno_color)
ax.plot(deno_6_x, deno_y, deno_6_z, linewidth=0.5, alpha=0.7, zorder=3, color=deno_color)

ax.plot([-7, 7], [0, 0], [7, 7], color='#cccccc', linewidth=0.6, zorder=10)
ax.plot([7, 7], [0, 0], [-7, 7], color='#cccccc', linewidth=0.6, zorder=10)

ax.plot([-7, 7], [1, 1], [7, 7], color='#cccccc', linewidth=0.6, zorder=10)
ax.plot([7, 7], [1, 1], [-7, 7], color='#cccccc', linewidth=0.6, zorder=10)

ax.plot([-7, 7], [1.2, 1.2], [7, 7], color='#cccccc', linewidth=0.6, zorder=10)
ax.plot([7, 7], [1.2, 1.2], [-7, 7], color='#cccccc', linewidth=0.6, zorder=10)

ax.plot([-7, 7], [2.2, 2.2], [7, 7], color='#cccccc', linewidth=0.6, zorder=10)
ax.plot([7, 7], [2.2, 2.2], [-7, 7], color='#cccccc', linewidth=0.6, zorder=10)

ax.plot([-7, 7], [0, 0], [0, 0], '--', color='#bbb8b8', linewidth=0.5)
ax.plot([0, 0], [0, 0], [-7, 7], '--', color='#bbb8b8', linewidth=0.5)

ax.plot([-7, 7], [2.2, 2.2], [0, 0], '--', color='#bbb8b8', linewidth=0.5)
ax.plot([0, 0], [2.2, 2.2], [-7, 7], '--', color='#bbb8b8', linewidth=0.5)


ax.set(xlim=[-7, 7], ylim=[-0.1, 2.3], zlim=[-7, 7])
ax.set(xticks=[], zticks=[], yticks=[0, 1, 1.2, 2.2])
ax.set_yticklabels([0,3,3,0], fontsize=6)

ax.set_box_aspect([1, 4, 1])
ax.view_init(32, -35)

plt.show()