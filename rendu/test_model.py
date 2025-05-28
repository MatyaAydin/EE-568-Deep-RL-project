import gymnasium as gym
from stable_baselines3 import PPO, TD3, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse

ENVS = {
    'pendulum': 'Pendulum-v1',
    'mountaincar': 'MountainCarContinuous-v0',
    'cartpole': 'CartPole-v1',
    'acrobot': 'Acrobot-v1'
}

algorithms = {
    'td3': TD3,
    'sac': SAC,
    'ppo': PPO,
    'dqn': DQN
}

def make_env(env_id, render_mode=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model testing script')
    parser.add_argument('--env', type=str, choices=['pendulum', 'mountaincar', 'cartpole', 'acrobot'], 
                       default='pendulum', help='Environment to optimize for')
    parser.add_argument('--algorithm', type=str, choices=['td3', 'sac', 'ppo', 'dqn'], 
                       default='td3', help='RL algorithm to optimize')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()
    env_name = ENVS[args.env]

    algo = algorithms[args.algorithm]

    model_path = args.model_path

    env = DummyVecEnv([make_env(env_name, render_mode="human")])
    model = algo.load(model_path, env=env)
    model.set_env(env)

    # Run the model
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
    print(f"Total reward: {total_reward[0]:.2f}")

    env.close()