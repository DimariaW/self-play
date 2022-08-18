import random
import enum
import gym
import numpy as np
import collections
from typing import Tuple, Dict, Any, Optional
from gfootball.env.observation_preprocessing import generate_smm

import rl.agent as agent
import rl.utils as utils


class Action(enum.IntEnum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass = 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18


sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]

action_to_sticky_index = {
    a: index for index, a in enumerate(sticky_index_to_action)
}


class SMMActionMaskWrapper(gym.Wrapper):
    def __init__(self, env, use_match_state=False):
        super().__init__(env)
        self.smm_obs = collections.deque(maxlen=4)
        self.use_match_state = use_match_state

    @staticmethod
    def illegal_actions(observations):
        illegal_action_mask = np.zeros((len(observations), 19))
        for i, obs in enumerate(observations):
            player_num = obs['active']
            player_pos_x, player_pos_y = obs['left_team'][player_num]
            ball_x, ball_y, ball_z = obs['ball']
            ball_x_relative = ball_x - player_pos_x
            ball_y_relative = ball_y - player_pos_y
            ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])

            illegal_actions = list()
            ball_owned_team = obs['ball_owned_team']
            if ball_owned_team == 1:  # opponent owned
                illegal_actions.append(int(Action.LongPass))
                illegal_actions.append(int(Action.HighPass))
                illegal_actions.append(int(Action.ShortPass))
                illegal_actions.append(int(Action.Shot))
                illegal_actions.append(int(Action.Dribble))
            elif ball_owned_team == -1:  # free
                illegal_actions.append(int(Action.Dribble))
            elif ball_owned_team == 0:  # owned
                illegal_actions.append(int(Action.Slide))

            if ball_distance >= 0.03:
                illegal_actions.append(int(Action.LongPass))
                illegal_actions.append(int(Action.HighPass))
                illegal_actions.append(int(Action.ShortPass))
                illegal_actions.append(int(Action.Shot))
                illegal_actions.append(int(Action.Dribble))
                illegal_actions.append(int(Action.Slide))

            # Already sticky action?
            sticky_actions = obs['sticky_actions']

            if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
                illegal_actions.append(int(Action.ReleaseSprint))

            if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
                illegal_actions.append(int(Action.ReleaseDribble))

            if 1 not in sticky_actions[:8]:
                illegal_actions.append(int(Action.ReleaseDirection))

            illegal_action_mask[[i] * len(illegal_actions), illegal_actions] = 1
        return illegal_action_mask

    def reset(self):
        raw_observations = self.env.reset()
        smm_obs = generate_smm(raw_observations)
        illegal_action_mask = self.illegal_actions(raw_observations)
        self.smm_obs.extend([smm_obs] * 4)
        smm_obs = np.concatenate(self.smm_obs, axis=-1)
        smm_obs = np.transpose(smm_obs, axes=(0, 3, 1, 2))
        if self.use_match_state:
            match_state = self.get_match_state(raw_observations)
            smm_obs = np.concatenate([smm_obs, match_state], axis=1)
        return {"smm": smm_obs, "mask": illegal_action_mask}

    def step(self, actions):
        raw_observations, reward, done, info = self.env.step(actions)
        smm_obs = generate_smm(raw_observations)
        illegal_action_mask = self.illegal_actions(raw_observations)
        self.smm_obs.append(smm_obs)
        smm_obs = np.concatenate(self.smm_obs, axis=-1)
        smm_obs = np.transpose(smm_obs, axes=(0, 3, 1, 2))
        if self.use_match_state:
            match_state = self.get_match_state(raw_observations)
            smm_obs = np.concatenate([smm_obs, match_state], axis=1)
        return {"smm": smm_obs, "mask": illegal_action_mask}, reward, done, info

    @staticmethod
    def get_match_state(obs):
        match_state = np.zeros((len(obs), 5, 72, 96))
        for index, observation in enumerate(obs):
            steps_left = observation["steps_left"]
            match_state[index, 0, :, :] = steps_left / 3001 * 255
            score = observation["score"][0]
            score_diff = observation["score"][0] - observation["score"][1]
            match_state[index, 1, :, :] = score / 10 * 255
            match_state[index, 2, :, :] = (score_diff + 10) / 20 * 255

            for ind, pos in enumerate(observation["left_team"]):
                x, y = SMMActionMaskWrapper.get_points(pos)
                match_state[index, 3, y, x] = observation["left_team_tired_factor"][ind] * 255

            for ind, pos in enumerate(observation["right_team"]):
                x, y = SMMActionMaskWrapper.get_points(pos)
                match_state[index, 4, y, x] = observation["right_team_tired_factor"][ind] * 255
        return match_state

    @staticmethod
    def get_points(point):
        x = int((point[0] + 1) / 2 * 96)
        y = int((point[1] + 1/2.25) / (2/2.25) * 72)
        x = max(0, min(96 - 1, x))
        y = max(0, min(72 - 1, y))
        return x, y


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, use_match_state=False):
        env = SMMActionMaskWrapper(env, use_match_state=use_match_state)
        super(EnvWrapper, self).__init__(env)
        self.use_match_state = use_match_state

    def reset(self):
        obs = self.env.reset()
        for _ in range(random.randint(0, 100)):
            obs, reward, done, info = self.env.step([0])
        return utils.get_element_from_batch(obs, 0)

    def step(self, action) -> Tuple[Any, Dict, bool, bool, Dict]:
        obs, reward, done, info = self.env.step([action])
        reward_infos = {"checkpoints": reward, "scoring": info["score_reward"]}
        truncated = False
        if done and not self.use_match_state:
            truncated = True
            done = False
        return utils.get_element_from_batch(obs, 0), reward_infos, done, truncated, info


class OpponentWrapper(gym.Wrapper):
    def __init__(self, env, opponents_pool: Dict[str, agent.Agent]):
        super().__init__(env)

        self.opponents_pool = opponents_pool
        self.opponent_agent = None
        self.opponent_obs = None

        self.scoring = 0

    def reset(self, model_id: Tuple[str, Optional[int]], weights: Optional) -> np.ndarray:
        model_name, model_index = model_id
        self.opponent_agent = self.opponents_pool[model_name]
        self.opponent_agent.set_weights(weights, model_index)

        obs = self.env.reset()
        for _ in range(random.randint(0, 100)):
            obs, reward, done, info = self.env.step([0, 0])
        self.opponent_obs = utils.get_element_from_batch(obs, 1)
        self.scoring = 0
        return utils.get_element_from_batch(obs, 0)

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, float], bool, Dict]:
        opponent_action = self.opponent_agent.predict(utils.to_numpy(self.opponent_obs, unsqueeze=0))["action"][0]

        obs, reward, done, info = self.env.step([action] + [opponent_action])

        obs: np.ndarray
        reward: np.ndarray
        done: bool
        info: dict

        reward_infos = {"checkpoints": reward[0], "scoring": info["score_reward"]}

        self.opponent_obs = utils.get_element_from_batch(obs, 0)
        self.scoring += info["score_reward"]

        if done:
            info["win"] = int(self.scoring > 0)
            info["opponent_id"] = self.opponent_agent.model_id

        return utils.get_element_from_batch(obs, 0), reward_infos, done, info


if __name__ == "__main__":
    import gfootball.env as gfootball_env
    from tests.football.football_model import CNNModel
    from rl.agent import IMPALAAgent
    import tqdm
    utils.set_process_logger()
    env_ = gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                            render=False,
                                            representation="raw",
                                            rewards="scoring,checkpoints")
    env_ = EnvWrapper(env_, use_match_state=True)
    model = CNNModel((21, 72, 96))
    agent = IMPALAAgent(model)
    obs_ = env_.reset()

    timeit = tqdm.tqdm()
    while True:
        timeit.update()
        action_ = agent.sample(utils.to_numpy(obs_, unsqueeze=0))["action"][0]
        obs_, reward_infos_, done_, truncated_, info_ = env_.step(action_)
        if done_ or truncated_:
            env_.reset()
