from env_models import football
from rl.agent import PPOAgent
from rl.actor import ActorEvaluator
import pickle
import rl.utils as utils


def test_vs_hard_ai():
    utils.set_process_logger()
    env_model_config = football.ENV_MODELS["multi_agent_academy_3_vs_1_with_keeper"]
    env = env_model_config["env"]()
    model = env_model_config["model"]()
    agent = PPOAgent(model)
    weights = pickle.load(open("./feature_80000.pickle", "rb"))
    agent.set_weights(weights, 36000)
    agents_pool = {agent.model_id[0]: agent}
    actor = ActorEvaluator(env, agents_pool, num_episodes=10, process_bar=True)

    actor.reset_agent(("feature", 36000), weights)
    actor.reset_env()
    env.render()
    for cmd, info in actor.predict():
        print(info)
        actor.reset_env()


def test_vs_tamak():
    utils.set_process_logger()
    env_model_config = football.ENV_MODELS["vs_tamak"]
    env = env_model_config["env"]()
    model = env_model_config["model"]()
    agent = PPOAgent(model)
    weights = pickle.load(open("./env_models/football/weights/feature_vs_tamak_70.pickle", "rb"))
    agent.set_weights(weights, 32000)
    agents_pool = {agent.model_id[0]: agent}
    actor = ActorEvaluator(env, agents_pool, num_episodes=10, process_bar=True)

    actor.reset_agent(("feature", 32000), weights)
    actor.reset_env()
    win_num = 0
    for cmd, info in actor.predict():
        print(info)
        win_num += info["meta_info"]["win"]
        actor.reset_env()
    print(win_num)


def test_battle():
    import copy
    import logging
    utils.set_process_logger()
    env_model_config = football.ENV_MODELS["battle"]
    env = env_model_config["env"]()

    models_pool = {
        ("builtin_ai", None): None,
        ("tamak", 1679): pickle.load(open("./env_models/football/weights/tamak_1679.pickle", "rb")),
        ("feature", 1): pickle.load(open("./tests/test_env_models/feature_vs_tamak_60.pickle", "rb")),
        ("feature", 2): pickle.load(open("./tests/test_env_models/feature_vs_tamak_70.pickle", "rb")),
        ("feature", 3): pickle.load(open("./tests/test_env_models/feature_200000.pickle", "rb"))
    }

    actor = ActorEvaluator(env, agents_pool=copy.deepcopy(env.agents_pool), num_episodes=10, process_bar=True)

    all_models = [model_id for model_id in models_pool]
    # model_id = random.choice(all_models)
    # opponent_id = random.choice(all_models)
    model_id = ("feature", 2)
    opponent_id = ("tamak", 1679)
    # opponent_id = ("feature", 2)
    # opponent_id = ("builtin_ai", None)
    # model_id, opponent_id = opponent_id, model_id
    logging.info(f"{model_id}, {opponent_id}")

    actor.reset_agent(model_id, models_pool[model_id])
    actor.reset_env(model_id, opponent_id, models_pool[opponent_id])

    # env.render()
    win_num = 0
    for cmd, info in actor.predict():
        print(info)
        win_num += info["win"]
        # model_id, opponent_id = opponent_id, model_id
        actor.reset_agent(model_id, models_pool[model_id])
        actor.reset_env(model_id, opponent_id, models_pool[opponent_id])
    print(win_num)


if __name__ == "__main__":
    # test_vs_hard_ai()
    test_vs_tamak()
    # test_battle()
