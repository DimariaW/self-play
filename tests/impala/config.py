from tests.env_models import ENV_MODEL_CONFIG

CONFIG = [
    {
        "name": "cartpole-fc-128-steps-3000_no_priority-impala_behavior_mse",
        "env_model_config": ENV_MODEL_CONFIG["cartpole-fc"],
        "num_steps": 128,
        "get_full_episodes": False,
        "use_bz2": True,
        "compressed_step_length": 16,
        "self-play": False,
        "memory_type": "list",
        "critic_update_method": "behavior",
        "upgo_key": (),
        "metrics_dir": "./log/cartpole/"
    },
    {
        "name": "cartpole-fc-32_steps-queue-impala_target_mse",
        "env_model_config": ENV_MODEL_CONFIG["cartpole-fc"],
        "num_steps": 32,
        "get_full_episodes": False,
        "use_bz2": True,
        "compressed_step_length": "full",
        "self-play": False,
        "memory_type": "queue",
        "critic_update_method": "target",
        "upgo_key": (),
        "metrics_dir": "./log/cartpole/"
    },
    {
        "name": "lunarlander-fc-full_episodes-3000_no_priority-impala_behavior_mse",
        "env_model_config": ENV_MODEL_CONFIG["lunarlander-fc"],
        "num_steps": 32,
        "get_full_episodes": True,
        "use_bz2": True,
        "compressed_step_length": 16,
        "self-play": False,
        "memory_type": "list",
        "critic_update_method": "behavior",
        "upgo_key": (),
        "metrics_dir": "./log/lunarlander/",
        "sleep_seconds": 0,
    },
    {
        "name": "lunarlander-fc-32_steps-queue-impala_target_mse",
        "env_model_config": ENV_MODEL_CONFIG["lunarlander-fc"],
        "num_steps": 32,
        "get_full_episodes": False,
        "use_bz2": True,
        "compressed_step_length": "full",
        "self-play": False,
        "memory_type": "queue",
        "critic_update_method": "target",
        "upgo_key": (),
        "metrics_dir": "./log/lunarlander/",
        "sleep_seconds": 0,
    },

][3]


LEAGUE_ADDRESS = ("172.18.236.85", 7779)
MEMORY_ADDRESS = ("172.18.236.85", 7777)
MODEL_SERVER_ADDRESS = ("172.18.236.85", 7778)
