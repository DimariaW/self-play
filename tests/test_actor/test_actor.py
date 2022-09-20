
import torch

from rl.agent import IMPALAAgent, A2CAgent, PPOAgent
from rl.actor import ActorSampler, ActorEvaluator, Actor, PostProcess
from rl.utils import set_process_logger, batchify, to_tensor

from tests.env_models import CONFIG


def test_gae():
    import copy
    import time

    cfg = CONFIG["cartpole-fc"]
    env = cfg["env_class"](**cfg["env_args"])
    model = cfg["model_class"](**cfg["model_args"])
    agent = PPOAgent(model)
    sampler = ActorSampler(env, agents_pool={agent.model_id[0]: agent}, num_steps=320, get_full_episodes=False)

    sampler.reset_agent(agent.model_id, agent.get_weights())
    sampler.reset_env()

    post_process1 = Actor.get_cal_gae_function(0.99, 1)
    post_process2 = PostProcess.get_cal_gae_func(0.99, 1)

    for args, data in sampler.sample_generator():
        if args == "episodes":
            _, data = data
            data_copy = copy.deepcopy(data)
            beg1 = time.time()
            data1 = post_process1(data)
            beg2 = time.time()
            data2 = post_process2(data_copy)
            end = time.time()
            print(f"1:{beg2-beg1}, 2:{end-beg2}")
            for epi1, epi2 in zip(data1, data2):
                assert abs(epi1["adv_info"]["reward"] - epi2["adv_info"]["reward"]) < 1e-5
                assert abs(epi1["value_target_info"]["reward"] - epi2["value_target_info"]["reward"]) < 1e-5
            print("success")
        if args == "sample_infos":
            print(data)
            sampler.reset_env()


if __name__ == "__main__":
    set_process_logger()
    test_gae()






