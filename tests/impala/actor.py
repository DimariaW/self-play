import gym
import torch

import rl.core as core
from rl.agent import IMPALAAgent

from tests.impala.env import EnvWrapper
from tests.impala.model import Model


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self, gather_id: int, actor_id: int):
        env = gym.make("LunarLander-v2")
        env = EnvWrapper(env)
        device = torch.device("cpu")
        model = Model(8, 4, use_orthogonal_init=True, use_tanh=False).to(device)
        agent = IMPALAAgent(model, device)
        return env, agent
