
import math
from gymnasium import RewardWrapper

class TanhWrapper(RewardWrapper):
    def __init__(self, env, scan_steps: int = None):
        super().__init__(env)
        self.scan_steps = scan_steps
        self.counter = 0

        self.max_reward = - math.inf
    
    def _update_max(self, reward):
        if self.scan_steps is None or self.counter < self.scan_steps:           
            self.max_reward = max(abs(reward), self.max_reward)
            self.counter += 1

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._update_max(reward)    
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        return math.tanh(reward / self.max_reward)