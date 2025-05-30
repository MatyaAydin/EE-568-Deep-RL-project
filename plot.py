import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set_theme()


# TODO put all envs
ENV_NAME = {"pendulum": "Pendulum-v1",
            "mountain_car_cont": "MountainCar-continuous-v0"
            }

def plot_all_algo(folders):
    fig, ax = plt.subplots(1, len(folders), figsize=(8 * len(folders), 5), squeeze=False)
    plt.suptitle("Reward over steps for different algorithms on each environment")
    for i, folder_name in enumerate(folders):
        for f in os.listdir(folder_name):
            algo_name = f[:-4] # ignore .txt

            data = np.loadtxt(os.path.join(folder_name, f), delimiter=",")
            steps = data[:, 0]
            mean_reward = data[:, 1]
            std_reward = data[:, 2]

            ax[0, i].plot(steps, mean_reward, label=algo_name)
            ax[0, i].fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)

        ax[0, i].set_title(f"{ENV_NAME[folder_name]}")
        ax[0, i].set_xlabel("Steps")
        ax[0, i].set_ylabel("Reward")
        ax[0, i].legend(loc='lower right')

    plt.tight_layout()
    #plt.savefig("all_algo.pdf")
    plt.show()


# TODO put all envs
plot_all_algo(["pendulum", "pendulum"])
