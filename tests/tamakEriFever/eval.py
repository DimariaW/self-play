from tests.tamakEriFever.actor import ActorMain
import numpy as np
import rl.utils as utils
import logging
from tqdm import tqdm

utils.set_process_logger()
env, agent = ActorMain().create_env_and_agent()
env.render()
#agent.set_weights(np.load("./easy_model/model_346346.npy", allow_pickle=True).item())
agent.set_weights(np.load("./hard_model/model_80031.npy", allow_pickle=True).item())

rewards = []
reward = 0
obs = env.reset()

time_it = tqdm()
while True:
    time_it.update()
    action = agent.predict(utils.batchify([obs], unsqueeze=0))["action"][0]
    obs, rew, done, info = env.step(action)
    reward += rew
    if done:
        obs = env.reset()
        rewards.append(reward)
        reward = 0

