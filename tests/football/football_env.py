import random

import gfootball.env as gfootball_env
import gym
import numpy as np
from typing import Tuple, Dict

import rl.agent as agent


class BuiltinAI(agent.Agent):
    def predict(self, obs: np.ndarray):
        return {"action": np.array([19] * len(obs))}

    def sample(self, *args, **kwargs):
        pass

    def set_weights(self, weights=None, model_index=None):
        pass

    def get_weights(self):
        pass

    @property
    def model_id(self):
        return "builtin_ai", 0


class OpponentWrapper(gym.Wrapper):
    def __init__(self, env, opponents_pool: Dict[str, agent.Agent]):
        super().__init__(env)

        self.opponents_pool = opponents_pool
        self.opponent_agent = None
        self.opponent_obs = None

        self.scoring = 0

    def reset(self, model_id: Tuple[str, int], weights) -> np.ndarray:
        model_name, model_index = model_id
        self.opponent_agent = self.opponents_pool[model_name]
        self.opponent_agent.set_weights(weights, model_index)

        obs: np.ndarray = self.env.reset()  # shape(2, ...)
        for _ in range(random.randint(0, 100)):
            obs, reward, done, info = self.env.step([0, 0])
        self.opponent_obs = obs[1:, ...]  # batch_size = 1

        self.scoring = 0

        return obs[0, ...]

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, float], bool, Dict]:
        opponent_action = self.opponent_agent.predict(self.opponent_obs)["action"][0]

        obs, reward, done, info = self.env.step([action] + [opponent_action])

        obs: np.ndarray
        reward: np.ndarray
        done: bool
        info: dict

        reward_infos = {"checkpoints": reward[0], "scoring": info["score_reward"]}

        self.opponent_obs = obs[1:, ...]  # batchified
        self.scoring += info["score_reward"]

        if done:
            info["win"] = int(self.scoring > 0)
            info["opponent_id"] = self.opponent_agent.model_id

        return obs[0, ...], reward_infos, done, info


class CHWWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.transpose(observation, (0, 3, 1, 2))



