
import gym
from typing import Tuple, Dict, Any


class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.model_id = None

    def reset(self, model_id, model_weights):
        self.model_id = model_id
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, Dict[str, float], bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        if done:
            info["opponent_id"] = self.model_id
            info["win"] = 1
        return obs, {"reward": reward/200}, done, info
