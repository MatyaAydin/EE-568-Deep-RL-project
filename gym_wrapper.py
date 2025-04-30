import gym

class GymEnvWrapper:
    """Standardized environment interface"""
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0] \
            if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def get_state_dim(self):
        return self.state_dim
    
    def get_action_dim(self):
        return self.action_dim