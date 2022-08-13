import bz2
import pickle

import gfootball.env as gfootball_env
from tests.tamakEriFever.football_env import TamakEriFeverEnv
from tests.tamakEriFever.football_model import FootballNet
from rl.agent import IMPALAAgent

import rl.core as core
import torch


class ActorMain(core.ActorMainBase):
    def create_env_and_agent(self, gather_id: int = None, actor_id: int = None):
        env = gfootball_env.create_environment(env_name="11_vs_11_easy_stochastic",
                                               rewards="scoring,checkpoints",
                                               render=False,
                                               representation="raw")
        env = TamakEriFeverEnv(env)
        device = torch.device("cpu")
        model = FootballNet().to(device)
        agent = IMPALAAgent(model, device)
        return env, agent


"""
if __name__ == "__main__":
    from rl.actor import Actor
    env, agent = ActorMain().create_env_and_agent()
    actor = Actor(env, agent, 32, get_full_episode=False)
    episodes = []
    episodes_compressed = []
    for _ in range(4):
        episode = actor.sample()
        episodes.append(episode)
        episodes_compressed.append(bz2.compress(pickle.dumps(episode)))
    assert(True)
"""
