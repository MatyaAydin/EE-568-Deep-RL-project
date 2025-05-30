from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import gymnasium as gym
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, SAC, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch
import warnings
from torch.utils.tensorboard import SummaryWriter
import argparse
import json

warnings.filterwarnings('ignore')

ENV_NAMES = {
    'pendulum': 'Pendulum-v1',
    'mountaincar': 'MountainCarContinuous-v0',
    'cartpole': 'CartPole-v1',
    'acrobot': 'Acrobot-v1',
    'discrete_mountaincar': 'MountainCar-v0'
}

algorithms = {
    'td3': TD3,
    'sac-pendulum': SAC,
    'sac-mountaincar': SAC,
    'ppo': PPO,
    'dqn': DQN    
}

class MultiSeedEvalCallback(BaseCallback):
    """
    Custom callback that evaluates the model on multiple seeds and averages the results.
    """
    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        n_eval_seeds: int = 3,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.n_eval_seeds = n_eval_seeds
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

        if self.log_path is not None:
            os.makedirs(log_path, exist_ok=True)
            self.log_file = open(os.path.join(log_path, "eval_logs.txt"), "w")

        if self.best_model_save_path is not None:
            os.makedirs(best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate on multiple seeds
            seed_rewards = []
            for seed in range(self.n_eval_seeds):
                rewards = []
                for _ in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    self.eval_env.seed(seed)
                    done = False
                    episode_reward = 0.0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                        obs, reward, done, info = self.eval_env.step(action)
                        episode_reward += reward
                    rewards.append(episode_reward)

                seed_mean_reward = np.mean(rewards)
                seed_rewards.append(seed_mean_reward)

            # Calculate average across seeds
            mean_reward = np.mean(seed_rewards)
            std_reward = np.std(seed_rewards)

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

            if self.log_path is not None:
                self.log_file.write(f"{self.num_timesteps},{mean_reward},{std_reward}\n")
                self.log_file.flush()

            if mean_reward > self.best_mean_reward and self.best_model_save_path is not None:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} - Saving model")
                self.model.save(os.path.join(self.best_model_save_path, "best_model"), include = ["env"])
                

        return True

    def _on_training_end(self) -> None:
        if self.log_path is not None:
            self.log_file.close()



def evaluate_model(model, vec_env, n_episodes=10):
    """
    Evaluate a RL model on a DummyVecEnv over multiple seeds.
    :param model: (BaseAlgorithm) the RL agent
    :param vec_env: (DummyVecEnv) the vectorized gym environment
    :param n_episodes: (int) number of episodes to evaluate
    :return: (float) mean reward
    """
    rewards = []
    for i in range(3):  # 3 seeds
        vec_env.seed(i)
        episode_rewards = []

        for j in range(n_episodes):
            obs = vec_env.reset()  # Returns array shape (1, obs_dim)
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                # reward and dones are arrays with one element each
                episode_reward += float(reward[0])
                done = bool(dones[0])
            episode_rewards.append(episode_reward)
        rewards.append(np.mean(episode_rewards))
    return np.mean(rewards)


def objective(trial):
    with open("algorithm_hyperparams.json", "r") as f:
        hyperpameters = json.load(f)
        hyperparams_dict = hyperpameters[algorithm]
        hyperparams = {}
        for key, value in hyperparams_dict.items():
            if key == "policy_kwargs":
                hyperparams[key] = {
                    'net_arch': trial.suggest_categorical('net_arch', value['net_arch'])
                }
            else:
                if value[0] == "categorical":
                    hyperparams[key] = trial.suggest_categorical(key, value[1])
                elif value[0] == "float":
                    hyperparams[key] = trial.suggest_float(key, value[1], value[2], log=True)
                elif value[0] == "int":
                    hyperparams[key] = trial.suggest_int(key, value[1], value[2])
    



    trial_log_dir = f"./logs/{algorithm}_{env_name}/trial_{trial.number}"
    os.makedirs(trial_log_dir, exist_ok=True)

    with open(f"{trial_log_dir}/hyperparams.json", "w") as f:
        json.dump({
            'params': hyperparams,
            'trial_number': trial.number
        }, f, indent=4)
    env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    if "noise_type" in hyperparams:
        noise_type = hyperparams["noise_type"]
        noise_std = hyperparams.get("noise_std")
        if noise_type == 'normal':
            action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                           sigma=noise_std * np.ones(env.action_space.shape))
        else:
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape),
                                                       sigma=noise_std * np.ones(env.action_space.shape))
        hyperparams_without_noise = {k: v for k, v in hyperparams.items() 
                                   if k not in ["noise_type", "noise_std"]}
        hyperparams = hyperparams_without_noise
        hyperparams["action_noise"] = action_noise

    print(f"Environment: {env_name}")
    model = algorithm_class(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=f"./logs/{algorithm}_{env_name}/trial_{trial.number}",
        **hyperparams
    )

    eval_callback = MultiSeedEvalCallback(
        eval_env=env,
        n_eval_episodes=5,          
        eval_freq=1000,             
        n_eval_seeds=3,             
        log_path=f"./logs/{algorithm}_{env_name}/trial_{trial.number}/",
        best_model_save_path=f"./logs/{algorithm}_{env_name}/trial_{trial.number}/",
        deterministic=True,
        verbose=1
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        mean_reward = evaluate_model(model, env, n_episodes=10)
        folders = os.listdir(f"./logs/{algorithm}_{env_name}/trial_{trial.number}/")
        folders = [folder for folder in folders if os.path.isdir(os.path.join(
            f"./logs/{algorithm}_{env_name}/trial_{trial.number}/", folder))]
        folders = sorted(folders)
        final_folder = f"./logs/{algorithm}_{env_name}/trial_{trial.number}/{folders[-1]}"
        with open(f"{final_folder}/hyperparams.json", "w") as f:
            if "action_noise" in hyperparams:
                hyperparams["action_noise"] = str(hyperparams["action_noise"])
            json.dump({
                'params': hyperparams,
                'trial_number': trial.number
            }, f, indent=4)
    except Exception as e:
        print(f"Trial failed: {e}")
        mean_reward = -float('inf')

    del model
    env.close()
    return mean_reward

def optimize_hyperparams(algorithm, env_name, n_trials=50):

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=15)
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters
    with open(f"best_params_{algorithm}_{env_name}.txt", "w") as f:
        f.write(str(trial.params))

    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for RL algorithms')
    parser.add_argument('--env', type=str, choices=['pendulum', 'mountaincar', 'cartpole', 'acrobot', 'discrete_mountaincar'],
                       default='pendulum', help='Environment to optimize for')
    parser.add_argument('--algorithm', type=str, choices=['td3', 'sac-pendulum', 'sac-mountaincar', 'ppo', 'dqn'], 
                       default='td3', help='RL algorithm to optimize')
    parser.add_argument('--total_timesteps', type=int, default=5000,
                       help='Total timesteps for training the model')
    
    args = parser.parse_args()
    env = args.env


    global algorithm
    algorithm = args.algorithm
    global env_name
    env_name = ENV_NAMES[env]
    global total_timesteps
    total_timesteps = args.total_timesteps
    global algorithm_class
    algorithm_class = algorithms[algorithm]

    os.makedirs(f"./logs/{algorithm}_{env_name}", exist_ok=True)

    print(f"Optimizing {algorithm} for {env_name}...")
    study = optimize_hyperparams(algorithm, env_name, n_trials=10)
    print("Optimization completed.")