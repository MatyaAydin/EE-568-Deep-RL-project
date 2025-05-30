"""
Plot presented in the section further ablation studies of the SAC section.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set_theme()


def plot(values, path, title):
    for v in values:
        data = np.loadtxt(path + str(v) + "/sac_MountainCarContinuous-v0/trial_0/eval_logs.txt",
                          delimiter=",")
        mean = data[:, 1]
        std = data[:, 2]
        x = data[:, 0]  
        label =  r"$\alpha$" if title == "Entropy coefficient" else title
        plt.plot(x, mean, label=label+ "=" + str(v))
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(f"Rewards over time for different {title.lower()} on MountainCarContinuous") 
    plt.legend(loc='lower right')
    plt.savefig("./plots/" + title + ".pdf")
    plt.show()


ent_coefs = [0, 0.01, 0.1, 1, 10, 100, 'auto']
buff_sizes = [int(1e5), 5*int(1e4), int(1e6)]
train_freqs = [8, 16, 32, 64]

plot(ent_coefs, "./logs_ent_coef_", "Entropy coefficient")
plot(buff_sizes, "./logs_buffer_size_", "Buffer size")
plot(train_freqs, "./logs_train_freq_", "Training frequency")