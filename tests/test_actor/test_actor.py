
import torch
import gym

from rl.agent import IMPALAAgent, A2CAgent, PPOAgent
from rl.actor import Actor
from rl.utils import set_process_logger, batchify, to_tensor

from tests.env_models import cartpole

if __name__ == "__main__":
    set_process_logger()
    env = cartpole.Env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model1 = cartpole.Model()
    model1 = cartpole.ModelPseudoRNN()
    agent1 = IMPALAAgent(model1, device)

    actor = Actor(env, agent1, num_steps=32, get_full_episodes=True, num_episodes=1)
    actor.reset_env()

    for args, data in actor.sample():
        if args == "episodes":
            data = batchify([batchify(data, unsqueeze=0)], unsqueeze=0)
            data = to_tensor(data)
            output = model1(data["observation"])
        if args == "sample_infos":
            print(data)
            actor.reset_env()

    for args, data in actor.predict_generator():
        actor.reset_env()




