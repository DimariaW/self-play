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
    }
][1]

USE_BZ2 = True

LEAGUE_ADDRESS = ("172.18.237.99", 7779)
MEMORY_ADDRESS = ("172.18.237.99", 7777)
MODEL_SERVER_ADDRESS = ("172.18.237.99", 7778)
