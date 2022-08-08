import random

import gfootball.env as gfootball_env
from tests.football.football_env import CHWWrapper, OpponentWrapper, BuiltinAI
from tests.football.football_model import CNNModel
from rl.agent import IMPALAAgent
import rl.core as core
import torch


class ActorCreate(core.ActorCreateBase):
    def create_env_and_agent(self, gather_id: int = None, actor_id: int = None):

        env = gfootball_env.create_environment(env_name="11_vs_11_hard_stochastic",
                                               stacked=True,
                                               rewards="scoring,checkpoints",
                                               render=False,
                                               representation="extracted",
                                               number_of_left_players_agent_controls=1,
                                               number_of_right_players_agent_controls=1,
                                               other_config_options={"action_set": "v2"})
        env = CHWWrapper(env)

        opponents_pool = {}

        device = torch.device("cpu")
        model = CNNModel((16, 72, 96), 19, name="cnn").to(device)

        opponents_pool["cnn"] = IMPALAAgent(model, device=device)
        opponents_pool["builtin_ai"] = BuiltinAI()

        env = OpponentWrapper(env, opponents_pool)

        model = CNNModel((16, 72, 96), 19, name="cnn").to(device)
        agent = IMPALAAgent(model, device=device)
        return env, agent


if __name__ == "__main__":
    import numpy as np
    import rl.utils as utils
    import logging
    from rl.actor import Actor
    utils.set_process_logger()

    env_, agent_ = ActorCreate().create_env_and_agent()

    # actor_eval = Actor(env_, agent_, num_episodes=1)
    actor_sample = Actor(env_, agent_, num_steps=32)

    models_pool = [
        (("cnn", 0), np.load("./easy_model/model_346346.npy", allow_pickle=True).item()),
        (("cnn", 1), np.load("./hard_model/model_80031.npy", allow_pickle=True).item()),
        (("cnn", 2), np.load("./hard_model/model_742227.npy", allow_pickle=True).item()),
        (("builtin_ai", 0), None)
    ]

    model_id, weights = random.choice(models_pool)
    actor_sample.reset_env(model_id, weights)

    model_id, weights = random.choice(models_pool[:3])
    actor_sample.reset_agent(model_id, weights)

    # predict_generator = actor_eval.predict_generator()
    sample_generator = actor_sample.sample_generator()

    for cmd, msg in sample_generator:
        logging.info(f"cmd: {cmd}")

        if cmd == "sample_infos":
            logging.info(f"msg:{msg}")
            model_id, weights = random.choice(models_pool)
            actor_sample.reset_env(model_id, weights)

            model_id, weights = random.choice(models_pool[:3])
            actor_sample.reset_agent(model_id, weights)





