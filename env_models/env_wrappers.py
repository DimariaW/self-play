import gym
import numpy as np
from typing import Tuple, Dict, Any


class DiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=10):
        super().__init__(env)
        self.action_space = self.env.action_space
        self.lower_bound = self.action_space.low
        self.upper_bound = self.action_space.high
        self.bins = np.array(bins)
        self.bin_length = (self.upper_bound - self.lower_bound) / self.bins

    def action(self, action):
        return self.lower_bound + action*self.bin_length + 0.5*self.bin_length

    def reverse_action(self, action):
        pass

