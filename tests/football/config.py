from env_models import football

CONFIG = [  # name: env-model-actor-memory-alg
    {
        "name": "[vs_tamak]-[feature]-[full_episodes_rollout_length_16]-"
                "[3000_no_priority_batch_size_512]-[ppo_critic_behavior_update_gae_actor_dual_clip]",
        # env model
        "env_model_config": football.ENV_MODELS[["multi_agent_academy_3_vs_1_with_keeper", "vs_tamak"][1]],
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
        "batch_size": 512,
        # PPO
        "critic_update_method": ["behavior", "behavior_bootstrap", "target"][0],
        "using_critic_update_method_adv": True,
        "actor_update_method": ["naive", "standard", "dual_clip"][2],
        "max_update_num_per_seconds": 4,
        "metrics_dir": "./log/vs_tamak/"
    },
][0]


LEAGUE_ADDRESS = ("10.127.7.14", 7779)
MEMORY_ADDRESS = ("10.127.7.14", 7777)
MODEL_SERVER_ADDRESS = ("10.127.7.14", 7778)


