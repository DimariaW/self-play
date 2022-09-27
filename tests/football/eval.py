from tests.football.actor import ActorMain
import tests.football.config as cfg
import rl.actor as rl_actor
import rl.utils as utils

import logging
import pickle


utils.set_process_logger()
actor = ActorMain().create_actor_evaluator()

models_pool = {
        ("builtin_ai", None): None,
        ("tamak", 1679): pickle.load(open("./weights/tamak_1679.pickle", "rb")),
        ("feature", 360000): pickle.load(open("./weights/feature_360000.pickle", "rb")),
    }

all_models = [model_id for model_id in models_pool]

# model_id = random.choice(all_models)
# opponent_id = random.choice(all_models)

model_id = ("feature", 360000)

opponent_id = ("tamak", 1679)

logging.info(f"{model_id}, {opponent_id}")

actor.reset_agent(model_id, models_pool[model_id])
actor.reset_env(model_id, opponent_id, models_pool[opponent_id])
# actor.reset_env()
actor._actor.env.render()
for tag, data in actor.predict():
    if tag == "eval_infos":
        logging.info(data)

        # model_id = random.choice(all_models)
        # opponent_id = random.choice(all_models)

        # model_id, opponent_id = opponent_id, model_id

        # logging.info(f"{model_id}, {opponent_id}")

        actor.reset_agent(model_id, models_pool[model_id])
        actor.reset_env(model_id, opponent_id, models_pool[opponent_id])
        # actor.reset_env()
    elif tag == "sample_infos":
        logging.info(data)
        actor.reset_env(model_id, opponent_id, models_pool[opponent_id])


