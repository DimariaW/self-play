from tests.football.actor import ActorCreate
import numpy as np
import rl.utils as utils
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle
import bz2

utils.set_process_logger()
env, agent = ActorCreate().create_env_and_agent()
env.render()
#agent.set_weights(np.load("./easy_model/model_346346.npy", allow_pickle=True).item())
#agent.set_weights(np.load("./hard_model/model_742227.npy", allow_pickle=True).item())
compressed = np.load("./kaggle_model/model_370598.npy", allow_pickle=True).item()
agent.set_weights(pickle.loads(bz2.decompress(compressed)))

rewards_infos = defaultdict(list)
current_reward_info = defaultdict(lambda: 0)
obs = env.reset()

time_it = tqdm()
times = 0
while True:
    time_it.update()
    action = agent.predict(utils.batchify([obs], unsqueeze=0))["action"][0]
    obs, rew, done, info = env.step(action)
    for key, value in rew.items():
        current_reward_info[key] += value

    if done:
        times += 1
        obs = env.reset()
        for key, value in current_reward_info.items():
            rewards_infos[key].append(value)
        current_reward_info.clear()

        if times == 10:
            break


