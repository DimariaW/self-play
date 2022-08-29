
import gym
from typing import Tuple, Dict, Any
import logging
import torch
import torch.nn as nn
import rl.model as model



class LunarLanderEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make("LunarLander-v2")
        self.reward_threshold = 200
        super().__init__(env)
        """
        for LunarLander-v2, the threshold is 200, 
        for CartPole-v1, the threshold is 475
        """

    def reset(self):
        return self.env.reset()

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        truncated = info.get('TimeLimit.truncated', False)
        if done and truncated:
            done = False
        return obs, {"reward": reward / self.reward_threshold}, done, truncated, info


class DiscreteActionSpace(gym.ActionWrapper):
    def __init__(self, env, num_bins):
        super().__init__(env)
        self.num_action_head = env.action_space.shape[0]
        self.low_bound = env.action_space.low
        self.high_bound = env.action_space.high
        self.bin_length = (self.high_bound-self.low_bound) / num_bins
        self._action_space = gym.spaces.MultiDiscrete([num_bins] * self.num_action_head)

    def action(self, action):
        return (action + 0.5) * self.bin_length + self.low_bound

    def reverse_action(self, action):
        return (action - self.low_bound) // self.bin_length





class ModelMultiHead(model.ModelValueLogit):
    def __init__(self, state_dim: int, action_head: int, num_act: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1 + num_act*action_head)
        self.act_fn = nn.ReLU()
        self.num_act = num_act
        self.action_head = action_head

    def forward(self, obs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        h1 = self.act_fn(self.fc1(obs))
        h2 = self.act_fn(self.fc2(h1))
        output = self.fc3(h2)
        shape = output.shape
        value = output[..., 0]
        logits = output[..., 1:].view(*shape[:-1], self.action_head, self.num_act)
        return {"reward": value}, logits  # value and logit, value 的最后一维度需要squeeze


class Model(model.ModelValueLogit):
    def __init__(self, state_dim: int, num_act: int, use_orthogonal_init=True, use_tanh=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1 + num_act)
        if use_orthogonal_init:
            logging.info("use orthogonal init")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
        if use_tanh:
            logging.info("use tanh activate function")
            self.act_fn = nn.Tanh()
        else:
            self.act_fn = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        h1 = self.act_fn(self.fc1(obs))
        h2 = self.act_fn(self.fc2(h1))
        output = self.fc3(h2)

        return {"reward": output[..., 0]}, output[..., 1:]  # value and logit, value 的最后一维度需要squeeze








