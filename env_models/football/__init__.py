import os.path

import rl.agent as rl_agent
import rl.utils as utils
import numpy as np
from typing import Literal
import pickle

from .feature import Observation2Feature, FeatureModel
from .tamak import TamakEriFever, FootballNet, TamakAgent
from .envs import Env, FixOpponentEnv


class BuiltinAI(rl_agent.Agent):
    def __init__(self, name):
        self.name = name

    def predict(self, obs):
        return {"action": np.array([19] * utils.get_batch_size(obs))}

    def sample(self, *args, **kwargs):
        pass

    def set_weights(self, weights=None, model_index=None):
        pass

    def get_weights(self):
        pass

    @property
    def model_id(self):
        return self.name, None


class BuiltinAIMultiAgent(BuiltinAI):
    def predict(self, obs):
        num_agents = utils.get_batch_size(obs, axis=1)
        action = [19] * num_agents
        return {"action": np.array([action] * utils.get_batch_size(obs, axis=0))}


def create_agent_and_preprocess_func(name: Literal["feature", "tamak", "builtin_ai"]):
    if name == "feature":
        return rl_agent.PPOAgent(FeatureModel(name="feature")), Observation2Feature.preprocess_obs
    elif name == "tamak":
        return TamakAgent(FootballNet(name="tamak")), TamakEriFever.preprocess_obs
    elif name == "builtin_ai":
        return BuiltinAIMultiAgent(name="builtin_ai"), lambda obs, action_history: obs


class EnvModelFactory:

    @staticmethod
    def create_multi_agent_academy_3_vs_1_with_keeper():
        env = FixOpponentEnv("academy_3_vs_1_with_keeper", 3, 2)

        agent, preprocess_func = create_agent_and_preprocess_func("feature")
        ai, ai_preprocess_func = create_agent_and_preprocess_func("builtin_ai")
        env.add_agent("feature", agent,
                      process_func=lambda obs, action_history: preprocess_func(obs, action_history, 4, 2, 200))
        env.add_agent("builtin_ai", ai, process_func=ai_preprocess_func)
        env.set_opponent(("feature", None), ("builtin_ai", None), None)

        return env

    @staticmethod
    def create_vs_tamak():
        env = FixOpponentEnv("11_vs_11_kaggle", 1, 1)

        agent, preprocess_func = create_agent_and_preprocess_func("feature")
        tamak, tamak_preprocess_func = create_agent_and_preprocess_func("tamak")
        env.add_agent("feature", agent,
                      process_func=lambda obs, action_history: preprocess_func(obs, action_history, 11, 11, 1500))
        env.add_agent("tamak", tamak, process_func=tamak_preprocess_func)

        current_file_dir = os.path.dirname(__file__)
        weights_path = os.path.join(current_file_dir, "weights/tamak_1679.pickle")
        opponent_weights = pickle.load(open(weights_path, "rb"))
        env.set_opponent(("feature", None), ("tamak", 1679), opponent_weights)

        return env

    @staticmethod
    def create_vs_easy_ai():
        env = FixOpponentEnv("11_vs_11_easy_stochastic", 1, 1)

        agent, preprocess_func = create_agent_and_preprocess_func("feature")
        ai, ai_preprocess_func = create_agent_and_preprocess_func("builtin_ai")
        env.add_agent("feature", agent,
                      process_func=lambda obs, action_history: preprocess_func(obs, action_history, 11, 11, 1500))
        env.add_agent("builtin_ai", ai, process_func=ai_preprocess_func)
        env.set_opponent(("feature", None), ("builtin_ai", None), None)

        return env

    @staticmethod
    def create_vs_hard_ai():
        env = FixOpponentEnv("11_vs_11_kaggle", 1, 1)

        agent, preprocess_func = create_agent_and_preprocess_func("feature")
        ai, ai_preprocess_func = create_agent_and_preprocess_func("builtin_ai")
        env.add_agent("feature", agent,
                      process_func=lambda obs, action_history: preprocess_func(obs, action_history, 11, 11, 1500))
        env.add_agent("builtin_ai", ai, process_func=ai_preprocess_func)
        env.set_opponent(("feature", None), ("builtin_ai", None), None)

        return env

    @staticmethod
    def create_battle():
        env = Env("11_vs_11_kaggle", 1, 1)

        agent, preprocess_func = create_agent_and_preprocess_func("feature")
        ai, ai_preprocess_func = create_agent_and_preprocess_func("builtin_ai")
        tamak, tamak_preprocess_func = create_agent_and_preprocess_func("tamak")

        env.add_agent("feature", agent,
                      process_func=lambda obs, action_history: preprocess_func(obs, action_history, 11, 11, 1500))
        env.add_agent("builtin_ai", ai, process_func=ai_preprocess_func)
        env.add_agent("tamak", tamak, process_func=tamak_preprocess_func)

        return env


ENV_MODELS = {
    "multi_agent_academy_3_vs_1_with_keeper": {
        "env": EnvModelFactory.create_multi_agent_academy_3_vs_1_with_keeper,
        "model": lambda: FeatureModel(name="feature")
    },
    "vs_tamak": {
        "env": EnvModelFactory.create_vs_tamak,
        "model": lambda: FeatureModel(name="feature")
    },
    "vs_hard_ai": {
        "env": EnvModelFactory.create_vs_hard_ai,
        "model": lambda: FeatureModel(name="feature")
    },
    "vs_easy_ai": {
        "env": EnvModelFactory.create_vs_easy_ai,
        "model": lambda: FeatureModel(name="feature")
    },
    "battle": {
        "env": EnvModelFactory.create_battle,
        "model": lambda: FeatureModel(name="feature")
    }
}

__all__ = ["ENV_MODELS"]






