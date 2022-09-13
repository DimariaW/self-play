from tests.football.actor import ActorMain
import tests.football.config as cfg
import rl.actor as rl_actor
import rl.utils as utils

import logging
import pickle


utils.set_process_logger()
env, agents_pool = ActorMain().create_env_and_agents_pool()

# env.render()

actor = rl_actor.Actor(env, agents_pool, num_episodes=5, num_steps=32, get_full_episodes=False, process_bar=True)

models_pool = {
        ("builtin_ai", None): None,
        # ("tamak", 1679): utils.to_numpy(torch.load("./model_weights/tamak_1679.pth")),
        ("feature", 870000): pickle.load(open("./weights/feature_870000.pickle", "rb")),
    }

all_models = [model_id for model_id in models_pool]

# model_id = random.choice(all_models)
# opponent_id = random.choice(all_models)

model_id = ("feature", 870000)

# opponent_id = ("builtin_ai", None)

# logging.info(f"{model_id}, {opponent_id}")

actor.reset_agent(model_id, models_pool[model_id])
# actor.reset_env(model_id, opponent_id, models_pool[opponent_id])
actor.reset_env()

for tag, data in actor.sample():
    if tag == "eval_infos":
        logging.info(data)

        # model_id = random.choice(all_models)
        # opponent_id = random.choice(all_models)

        # model_id, opponent_id = opponent_id, model_id

        # logging.info(f"{model_id}, {opponent_id}")

        actor.reset_agent(model_id, models_pool[model_id])
        # actor.reset_env(model_id, opponent_id, models_pool[opponent_id])
        actor.reset_env()
    elif tag == "sample_infos":
        logging.info(data)


