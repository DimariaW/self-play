from tests.football.actor import ActorCreate
import rl.actor as rl_actor
import rl.utils as utils
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle
import bz2

utils.set_process_logger()
env, agent = ActorCreate().create_env_and_agent()
env.render()

actor = rl_actor.Actor(env, agent, num_episodes=10)

weights = pickle.load(open("./1_vs_1_model/cnn_132000.pickle", "rb"))
index = 132000

actor.reset_agent(("cnn", index), weights)
actor.reset_env(("builtin_ai", None), None)

for _, data in actor.predict():
    logging.info(data)
    actor.reset_env(("builtin_ai", None), None)


