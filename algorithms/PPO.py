import torch
import torch.nn as nn
import torch.optim as optim
from base import BaseAgent, ReplayBuffer


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization implementation"""
    def __init__(self, network, state_dim, action_dim, lr=3e-4, gamma=0.99, clip=0.2):
        self.actor = network(state_dim, action_dim)
        self.critic = network(state_dim, 1)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.buffer = ReplayBuffer(10000)
        
    def act(self, state, training=True):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action_mean = self.actor(state)
            value = self.critic(state)
        return action_mean.numpy(), value.item()
    
    def train(self, batch):
        # PPO-specific training logic
        pass  # Implementation similar to DQN but with clipped surrogate objective