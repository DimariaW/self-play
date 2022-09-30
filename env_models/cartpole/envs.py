import gym
from typing import Tuple, Any, Dict


class Env(gym.Wrapper):
    def __init__(self, name, reward_scale=1, **kwargs):
        env = gym.make(name, **kwargs)
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_unscaled = 0

    def reset(self):
        self.reward_unscaled = 0
        return self.env.reset()

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.reward_unscaled += reward
        truncated = info.get('TimeLimit.truncated', False)
        if done:
            info["eval_reward"] = self.reward_unscaled
            if truncated:
                done = False

        return obs, {"reward": reward/self.reward_scale}, done, truncated, info
