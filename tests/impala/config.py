from tests.env_models import ENV_MODEL_CONFIG

CONFIG = [
    {
        "name": "[cartpole]-[fc]-[full_episodes_rollout_length_16]-"
                "[3000_no_priority_batch_size_64]-[impala_critic_behavior_no_upgo]",
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
        "critic_update_method": ["behavior", "behavior_bootstrap", "target"][0],
        "use_upgo": False,
        "sleep_seconds": 0,
        "metrics_dir": "./log/cartpole/"
    },
    {
        "name": "[cartpole]-[fc]-[32_steps]-"
                "[queue_batch_size_16]-[impala_critic_target_no_upgo]",
        # env model
        "env_model_config": ENV_MODEL_CONFIG["cartpole-fc"],
        # actor
        "num_steps": 32,
        "get_full_episodes": False,
        "use_bz2": True,
        "compressed_step_length": "full",
        "self-play": False,
        # memory
        "memory_type": "queue",
        "maxlen": 32,
        "batch_size": 16,
        # PPO
        "critic_update_method": ["behavior", "behavior_bootstrap", "target"][2],
        "use_upgo": False,
        "sleep_seconds": 0,
        "metrics_dir": "./log/cartpole/"
    },

][0]


LEAGUE_ADDRESS = ("172.18.237.19", 7779)
MEMORY_ADDRESS = ("172.18.237.19", 7777)
MODEL_SERVER_ADDRESS = ("172.18.237.19", 7778)
