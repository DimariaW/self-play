import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from typing import Any, Tuple, Dict

from rl.agent import IMPALAAgent
from rl.model import ModelValueLogit
from rl.actor import Actor
from rl.utils import set_process_logger


class DuelNet(ModelValueLogit):
    def __init__(self, obs_dim, num_acts):
        super().__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, num_acts + 1)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        value_and_logits = self.fc3(h2)

        return {"reward": value_and_logits[..., 0]}, value_and_logits[..., 1:]


class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, Dict[str, float], bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        if done:
            info["opponent_id"] = ("test", 0)
            info["win"] = 0
        return obs, {"reward": reward}, done, info


if __name__ == "__main__":
    set_process_logger()
    env = gym.make("LunarLander-v2")
    env = EnvWrapper(env)

    obs_dim = env.observation_space.shape[0]
    num_acts = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelNet(obs_dim, num_acts).to(device)
    agent = IMPALAAgent(model, device, model_index=0)

    #actor = Actor(env, agent, num_steps=32, num_episodes=0, get_full_episode=False)
    #actor.reset_env()

    #for args, data in actor.sample_generator():
    #    print("----")
    #    if args == "sample_infos":
    #        actor.reset_env()

    actor = Actor(env, agent, num_steps=0, num_episodes=1, get_full_episode=False)
    actor.reset_env()

    for args, data in actor.predict_generator():
        actor.reset_env()






