from env_models import football

CONFIG = [  # name: env-model-actor-memory-alg
    {
        "name": "[vs_easy_ai]-[feature]-[full_episodes_rollout_length_16]-"
                "[3000_no_priority_batch_size_512]-[ppo_critic_behavior_update_gae_actor_dual_clip]",
        # env model
        "env_model_config": football.ENV_MODELS[
            ["multi_agent_academy_3_vs_1_with_keeper", "vs_tamak", "vs_easy_ai"][2]
        ],
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
        "using_critic_update_method_adv": True,
        "actor_update_method": ["naive", "standard", "dual_clip"][2],
        "max_update_num_per_seconds": 4,
        "metrics_dir": "./log/vs_easy_ai/"
    },  # PPO
    {
        "name": "[vs_easy_ai]-[feature]-[full_episodes_rollout_length_16]-"
                "[3000_no_priority_batch_size_64]-[impala_critic_behavior_upgo]",
        # env model
        "env_model_config": football.ENV_MODELS[
            ["multi_agent_academy_3_vs_1_with_keeper", "vs_tamak", "vs_easy_ai"][2]
        ],
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
        "max_update_num_per_seconds": 0,
        "metrics_dir": "./log/vs_easy_ai/"
    },  # impala-async_actor_learner
    {
        "name": "[vs_easy_ai]-[feature]-[32_steps]-[queue_batch_size_16]-[impala_critic_target_no_upgo]",
        # env model
        "env_model_config": football.ENV_MODELS[
            ["multi_agent_academy_3_vs_1_with_keeper", "vs_tamak", "vs_easy_ai"][2]
        ],
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
        "metrics_dir": "./log/vs_easy_ai/"
    },  # impala-sync_actor_learner
][0]


LEAGUE_ADDRESS = ("10.127.7.14", 7779)
MEMORY_ADDRESS = ("10.127.7.14", 7777)
MODEL_SERVER_ADDRESS = ("10.127.7.14", 7778)


