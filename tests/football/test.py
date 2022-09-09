def test_observation2feature():
    import gfootball.env as gfootball_env
    from tests.football.football_env import Observation2Feature
    env = gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                           render=False,
                                           representation="raw",
                                           rewards="scoring",
                                           number_of_left_players_agent_controls=1,
                                           number_of_right_players_agent_controls=1)
    env.reset()
    for _ in range(100):
        env.step([0, 0])
    obs, reward, info, done = env.step([0, 0])
    illegal_action_mask = Observation2Feature.create_illegal_action_masks(obs[0])
    ball_zone = Observation2Feature.encode_ball_which_zone(obs[0])
    ball_state, player_state = Observation2Feature.get_ball_and_player_feature(obs[0], ball_zone, illegal_action_mask)
    team_feature = Observation2Feature.get_team_feature(obs[0], 11, 11)
    print("---")


def test_feature_env_model():
    from tests.football.football_env import FeatureEnv
    from tests.football.feature_model import FeatureModel
    import rl.utils as utils

    env = FeatureEnv(reward_type="customized")
    model = FeatureModel()

    obs = env.reset()
    value_infos, logits = model(utils.to_tensor(utils.to_tensor(obs, unsqueeze=0), unsqueeze=0))

    next_obs, reward_infos, done, truncated, info = env.step(0)
    print("---")


def test_feature_env_4_multi_agent():
    from tests.football.football_env import FeatureEnv4MultiAgent
    from tests.football.feature_model import FeatureModel
    import rl.utils as utils
    import numpy as np

    env = FeatureEnv4MultiAgent()
    model = FeatureModel()

    obs = env.reset()
    value_infos, logits = model(utils.to_tensor(utils.to_tensor(obs, unsqueeze=0), unsqueeze=0))
    next_obs, reward_infos, done, truncated, info = env.step(np.array([0, 0, 0]))


def test_opponent_env():
    from tests.football.football_env import OpponentEnv
    from tests.football.feature_model import FeatureModel, BuiltinAI
    from rl.agent import IMPALAAgent
    import rl.utils as utils
    import pickle

    utils.set_process_logger()

    built_in_ai = BuiltinAI("ai")
    agent = IMPALAAgent(FeatureModel())
    opponents_pool = {built_in_ai.model_id[0]: built_in_ai, agent.model_id[0]: agent}
    env = OpponentEnv(opponents_pool)
    env.render()

    test_agent = IMPALAAgent(FeatureModel())
    weights = pickle.load(open("model_weights/feature_380000.pickle", "rb"))
    index = 380000
    test_agent.set_weights(weights, index)

    obs = env.reset(test_agent.model_id, test_agent.get_weights())
    while True:
        obs, reward_infos, done, truncated, info = env.step(test_agent.predict(utils.to_numpy(obs, unsqueeze=0))["action"][0])
        if done:
            print(info)


if __name__ == "__main__":
    # test_observation2feature()
    # test_feature_env_model()
    # test_feature_env_4_multi_agent()
    test_opponent_env()
