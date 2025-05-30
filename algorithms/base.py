import abc
from collections import deque, namedtuple
import random

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, *args):
        self.buffer.append(self.Transition(*args))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = self.Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.buffer)

class BaseAgent(abc.ABC):
    """Abstract base class for all RL agents"""
    @abc.abstractmethod
    def act(self, state, training=True):
        """Select action given current state"""
        pass
    
    @abc.abstractmethod
    def train(self, batch):
        """Update model parameters using training batch"""
        pass
    
    @abc.abstractmethod
    def save(self, path):
        """Save model weights"""
        pass
    
    @abc.abstractmethod
    def load(self, path):
        """Load model weights"""
        pass