from tests.env_models import ENV_MODEL_CONFIG

CONFIG = [  # name: env-model-actor-memory-alg
    {
        "name": "[cartpole]-[fc]-[full_episodes_rollout_length_16]-"
                "[3000_no_priority_batch_size_64]-[ppo_critic_behavior_update_gae_actor_standard]",
        # env model
        "env_model_config": ENV_MODEL_CONFIG["cartpole-fc"],
        # actor
        "num_steps": 32,
        "get_full_episodes": True,
        "use_bz2": True,
        "compressed_step_length": 16,
        "self-play": False,
        # memory
        "memory_type": "list",
        "maxlen": 3000,
        "priority_replay": False,
        "batch_size": 64,
        # PPO
        "critic_update_method": "behavior",
        "using_critic_update_method_adv": True,
        "actor_update_method": "standard",
        "sleep_seconds": 0,
        "metrics_dir": "./log/cartpole/"
    },
][0]


LEAGUE_ADDRESS = ("172.18.236.247", 7779)
MEMORY_ADDRESS = ("172.18.236.247", 7777)
MODEL_SERVER_ADDRESS = ("172.18.236.247", 7778)
