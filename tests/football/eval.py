from tests.football.actor import ActorMain
import rl.actor as rl_actor
import rl.utils as utils
import logging
import pickle

utils.set_process_logger()
env, agent = ActorMain().create_env_and_agent()
env.render()

actor = rl_actor.Actor(env, agent, num_episodes=10, num_steps=3000)

weights = pickle.load(open("./models/feature_110000.pickle", "rb"))
index = 110000

actor.reset_agent(("feature", index), weights)
# actor.reset_env(("builtin_ai", None), None)
actor.reset_env()

for tag, data in actor.sample_generator():
    if tag == "sample_infos":
        logging.info(data)
    # actor.reset_env(("builtin_ai", None), None)
        actor.reset_env()


