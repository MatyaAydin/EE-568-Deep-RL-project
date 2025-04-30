from base import BaseAgent, ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random





class DQNAgent(BaseAgent):
    """Deep Q-Network implementation"""
    def __init__(self, network, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.q_net = network(state_dim, action_dim)
        self.target_net = network(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 128
        self.action_dim = action_dim
        
    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()
    
    def train(self, batch):
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.q_net, self.target_net, tau=0.005)
        
        # Decay epsilon
        self.epsilon *= 0.995
        self.epsilon = max(self.epsilon, 0.01)
        return loss.item()