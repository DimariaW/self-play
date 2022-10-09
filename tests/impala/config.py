from env_models import cartpole

CONFIG = [
    {
        "name": "[MountainCarContinuous]-[FC]-[full_episodes_rollout_length_16]-"
                "[3000_no_priority_batch_size_64]-[impala_critic_behavior_upgo]",
        # env model
        "env_model_config": cartpole.ENV_MODELS[["CartPole-FC", "MountainCarContinuous-FC", "LunarLanderContinuous-FC"][0]],
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
        "use_upgo": True,
        "sleep_seconds": 0,
        "metrics_dir": "./log/MountainCarContinuous/"
    },
    {
        "name": "[CartPole]-[FC]-[32_steps]-[queue_batch_size_16]-[impala_critic_target_no_upgo]",
        # env model
        "env_model_config": cartpole.ENV_MODELS["CartPole-FC"],
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
        "max_update_num_per_seconds": 9999,
        "metrics_dir": "./log/cartpole/"
    },

][0]


LEAGUE_ADDRESS = ("192.168.43.157", 7779)
MEMORY_ADDRESS = ("192.168.43.157", 7777)
MODEL_SERVER_ADDRESS = ("192.168.43.157", 7778)
