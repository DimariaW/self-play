from tests.env_models import cartpole

CONFIG = [
    {
     "name": "cartpole-fc",
     "env_class": cartpole.Env,
     "env_args": {
         "name": "CartPole-v1",
         "reward_threshold": 475,
         },
     "model_class": cartpole.Model,
     "model_args": {
         "obs_dim": 4,
         "num_action": 2
         }
    },

    {
        "name": "cartpole-pseudo_rnn",
        "env_class": cartpole.Env,
        "env_args": {
            "name": "CartPole-v1",
            "reward_threshold": 475,
        },
        "model_class": cartpole.ModelPseudoRNN,
        "model_args": {
            "obs_dim": 4,
            "num_action": 2
        }
    },

    {
        "name": "cartpole-rnn",
        "env_class": cartpole.Env,
        "env_args": {
            "name": "CartPole-v1",
            "reward_threshold": 475,
        },
        "model_class": cartpole.ModelRNN,
        "model_args": {
            "obs_dim": 4,
            "num_action": 2
        }
    },

    {
        "name": "lunarlander-fc",
        "env_class": cartpole.Env,
        "env_args": {
            "name": "LunarLander-v2",
            "reward_threshold": 200,
        },
        "model_class": cartpole.Model,
        "model_args": {
            "obs_dim": 8,
            "num_action": 4
        },
    },

    {
        "name": "lunarlander-pseudo-rnn",
        "env_class": cartpole.Env,
        "env_args": {
            "name": "LunarLander-v2",
            "reward_threshold": 200,
        },
        "model_class": cartpole.ModelPseudoRNN,
        "model_args": {
            "obs_dim": 8,
            "num_action": 4
        }
    },

    {
        "name": "lunarlander-rnn",
        "env_class": cartpole.Env,
        "env_args": {
            "name": "LunarLander-v2",
            "reward_threshold": 200,
        },
        "model_class": cartpole.ModelRNN,
        "model_args": {
            "obs_dim": 8,
            "num_action": 4
        }
    },

    {
     "name": "cartpole-opponent_test",
     "env_class": cartpole.OpponentEnv,
     "env_args": {
         "name": "CartPole-v1",
         "reward_threshold": 475,
         },
     "model_class": cartpole.Model,
     "model_args": {
         "obs_dim": 4,
         "num_action": 2
         }
    }
][6]

USE_BZ2 = False

LEAGUE_ADDRESS = ("172.18.237.38", 7779)
MEMORY_ADDRESS = ("172.18.237.38", 7777)
MODEL_SERVER_ADDRESS = ("172.18.237.38", 7778)
