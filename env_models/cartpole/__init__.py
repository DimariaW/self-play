from ..env_wrappers import DiscreteWrapper
from .envs import Env
from .models import Model, ModelRNN, ModelPseudoRNN

VALID_ENV_NAMES = ["CartPole", "MountainCarContinuous",
                   "LunarLander", "BipedalWalker",
                   "BipedalWalker_hardcore"]

VALID_MODEL_NAMES = ["FC", "Sudo-RNN", "RNN"]


class EnvModelFactory:
    @staticmethod
    def create_cartpole():
        return Env("CartPole-v1", reward_scale=475)

    @staticmethod
    def create_mountaincarcontinuous():
        return DiscreteWrapper(Env("MountainCarContinuous-v0", reward_scale=50))

    @staticmethod
    def create_lunarlander():
        return Env("LunarLander-v2", reward_scale=200)

    @staticmethod
    def create_lunarlandercontinuous():
        return DiscreteWrapper(Env("LunarLander-v2", reward_scale=200, continuous=True), bins=10)

    @staticmethod
    def create_bipedalwalker():
        return DiscreteWrapper(Env("BipedalWalker-v3", reward_scale=300), bins=10)

    @staticmethod
    def create_fc(obs_dim, num_action, hidden_dim, action_heads):
        return Model(obs_dim, num_action, hidden_dim, action_heads)


ENV_MODELS = {
    "CartPole-FC": {
        "env": EnvModelFactory.create_cartpole,
        "model": lambda: EnvModelFactory.create_fc(4, 2, 128, 1),
    },
    "LunarLander-FC": {
        "env": EnvModelFactory.create_lunarlander,
        "model": lambda: EnvModelFactory.create_fc(8, 4, 128, 1),
    },
    "LunarLanderContinuous-FC": {
        "env": EnvModelFactory.create_lunarlandercontinuous,
        "model": lambda: EnvModelFactory.create_fc(8, 10, 128, 2),
    },
    "BipedalWalker-FC": {
        "env": EnvModelFactory.create_bipedalwalker,
        "model": lambda: EnvModelFactory.create_fc(24, 10, 128, 4),
    },
    "MountainCarContinuous-FC": {
        "env": EnvModelFactory.create_mountaincarcontinuous,
        "model": lambda: EnvModelFactory.create_fc(2, 10, 128, 1)
    }
}

__all__ = ["ENV_MODELS"]

