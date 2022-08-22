CONFIG = [
    {
     "env_name": "CartPole-v1",
     "reward_threshold": 475,
     "obs_dim": 4,
     "num_act": 2
    },
    {
     "env_name": "LunarLander-v2",
     "reward_threshold": 200,
     "obs_dim": 8,
     "num_act": 4
    },
    {
     "env_name": "LunarLander-v2",
     "continuous": True,
     "reward_threshold": 200,
     "obs_dim": 8,
     "action_head": 2,
     "num_bins": 10
    },
    {
     "env_name": "BipedalWalker-v3",
     "continuous": True,
     "reward_threshold": 300,
     "obs_dim": 24,
     "action_head": 4,
     "num_bins": 10
    }
][3]

USE_BZ2 = False

LEAGUE_ADDRESS = ("172.18.237.26", 7779)
MEMORY_ADDRESS = ("172.18.237.26", 7777)
MODEL_SERVER_ADDRESS = ("172.18.237.26", 7778)
