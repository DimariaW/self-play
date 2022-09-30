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
    weights = pickle.load(open("./feature_360000.pickle", "rb"))
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
    weights = pickle.load(open("./tests/test_env_models/feature_360000.pickle", "rb"))
    agent.set_weights(weights, 36000)
    agents_pool = {agent.model_id[0]: agent}
    actor = ActorEvaluator(env, agents_pool, num_episodes=10, process_bar=True)

    actor.reset_agent(("feature", 36000), weights)
    actor.reset_env()
    for cmd, info in actor.predict():
        print(info)
        actor.reset_env()


if __name__ == "__main__":
    test_vs_hard_ai()
    # test_vs_tamak()
