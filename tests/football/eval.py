from tests.football.actor import ActorMain
import rl.actor as rl_actor
import rl.utils as utils
import logging
import pickle
import torch
import random

utils.set_process_logger()
env, agents_pool = ActorMain().create_env_and_agent()
# env.render()

actor = rl_actor.Actor(env, agents_pool, num_episodes=4, get_full_episodes=True, process_bar=True)

models_pool = {
        ("builtin_ai", None): None,
        ("tamak", 1679): utils.to_numpy(torch.load("./model_weights/tamak_1679.pth")),
        ("feature", 126): pickle.load(open("./model_weights/feature_1260000.pickle", "rb")),
    }

all_models = [model_id for model_id in models_pool]

model_id = random.choice(all_models)
opponent_id = random.choice(all_models)

logging.info(f"{model_id}, {opponent_id}")

actor.reset_agent(model_id, models_pool[model_id])
actor.reset_env(model_id, opponent_id, models_pool[opponent_id])
# actor.reset_env()

for tag, data in actor.predict():
    if tag == "eval_infos":
        logging.info(data)
        model_id = random.choice(all_models)
        opponent_id = random.choice(all_models)

        logging.info(f"{model_id}, {opponent_id}")

        actor.reset_agent(model_id, models_pool[model_id])
        actor.reset_env(model_id, opponent_id, models_pool[opponent_id])


