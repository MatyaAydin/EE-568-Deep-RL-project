import abc
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np

from base_classes import BaseNetwork, RLTrainer
from gym_wrapper import GymEnvWrapper
from algorithms.DQN import DQNAgent
from algorithms.PPO import PPOAgent
from algorithms.TD3 import TD3Agent
from algorithms.SAC import SACAgent





    

if __name__ == "__main__":
    env = GymEnvWrapper("CartPole-v1")
    agent1 = DQNAgent(BaseNetwork, env.get_state_dim(), env.get_action_dim())
    agent2 = PPOAgent(BaseNetwork, env.get_state_dim(), env.get_action_dim())
    agent3 = TD3Agent(BaseNetwork, env.get_state_dim(), env.get_action_dim())
    agent4 = SACAgent(BaseNetwork, env.get_state_dim(), env.get_action_dim())
    agents = [agent1, agent2, agent3, agent4]
    max_episodes = [1000, 1000, 1000, 1000]
    trainer = RLTrainer(agents, env, max_episodes, plot=True, display=True)
    trainer.train()