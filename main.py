import abc
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np
from gym_wrapper import GymEnvWrapper
from algorithms.DQN import DQNAgent
from algorithms.PPO import PPOAgent
from algorithms.TD3 import TD3Agent
from algorithms.SAC import SACAgent
import matplotlib.pyplot as plt





class BaseNetwork(nn.Module):
    """Base neural network module"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)




class RLTrainer:
    """Generic training framework"""
    def __init__(self, agents, env, max_episodes):
        self.agents = agents
        self.env = env
        self.max_episodes = max_episodes
        if len(agents) != len(max_episodes):
            raise ValueError("agents and max_episodes must have the same length")

    def train(self, plot =False, display = False):
        losses = []
        for agent, max_episode in zip(self.agents, self.max_episodes):
            loss_list = []
            print(f"Training {agent.__class__.__name__} for {max_episode} episodes")
            for episode in range(max_episode):
                state = self.env.reset()
                episode_reward = 0
                
                while True:
                    action = agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    agent.buffer.add(state, action, reward, next_state, done)
                    episode_reward += reward
                    state = next_state
                    
                    if len(agent.buffer) > agent.batch_size:
                        batch = agent.buffer.sample(agent.batch_size)
                        loss = agent.train(batch)
                        loss_list.append(loss)  
                    
                    if done:
                        break
                
                print(f"Episode {episode} Reward: {episode_reward}")
            losses.append(loss_list)
        if plot:
            plt.figure(figsize=(10, 5))
            for i, loss_list in enumerate(losses):
                plt.plot(loss_list, label=f"Agent {i+1}")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title("Training Losses")
            plt.legend()
            plt.show()
        if display:
            self.env.render()

    

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