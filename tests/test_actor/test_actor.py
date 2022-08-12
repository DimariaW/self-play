
import torch
import gym

from rl.agent import IMPALAAgent, A2CAgent, PPOAgent
from rl.actor import Actor
from rl.utils import set_process_logger

from tests.test_env_models.env_model_gym import EnvWrapper, Model

if __name__ == "__main__":
    set_process_logger()
    env = gym.make("LunarLander-v2", max_episode_steps=100)
    env = EnvWrapper(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(8, 4).to(device)
    agent1 = IMPALAAgent(model, device)
    agent2 = A2CAgent(model, device)
    agent3 = PPOAgent(model, device)

    actor = Actor(env, agent2, num_steps=32, num_episodes=1)
    actor.reset_env()

    for args, data in actor.sample_generator():
        if args == "sample_infos":
            print(data)
            actor.reset_env()

    for args, data in actor.predict_generator():
        actor.reset_env()




