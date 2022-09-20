import tests.env_models.cartpole as cartpole


ENV_MODEL_CONFIG = {
    "cartpole-fc":
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

    "cartpole-pseudo_rnn":
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

    "cartpole-rnn":
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

    "lunarlander-fc":
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

    "lunarlander-pseudo-rnn":
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

    "lunarlander-rnn":
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

    "cartpole-opponent_test":
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
}

__all__ = [ENV_MODEL_CONFIG]

