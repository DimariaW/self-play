import random
import enum
import gym
import numpy as np
import collections
from typing import Tuple, Dict, Any, Optional, Literal
from gfootball.env.observation_preprocessing import generate_smm
import gfootball.env as gfootball_env
import rl.agent as agent
import rl.utils as utils

#%%


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


class BallZone(enum.IntEnum):
    PenaltyZone = 0
    BackZone = 1
    MidZone = 2
    FrontZone = 3
    ShotZone = 4
    OtherZone = 5


class GameMode(enum.IntEnum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6

ball_zone_to_reward = [
    -2, -1, 0, 1, 2, 0
]

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


class Observation2Feature:
    @staticmethod
    def _preprocess_obs(obs):
        mode = obs['game_mode']
        if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

            team_player_position = [(0, i, p) for i, p in enumerate(obs['left_team'])] + \
                                   [(1, i, p) for i, p in enumerate(obs['right_team'])]
            distances = [(t[0], t[1], dist(t[2], obs['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            # print(mode, [t[2] for t in distances])
            # print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            # input()
            obs['ball_owned_team'] = distances[0][0]
            obs['ball_owned_player'] = distances[0][1]
        return obs

    @staticmethod
    def create_illegal_action_masks(obs):
        illegal_action_set = set()
        player_num = obs['active']
        player_pos_x, player_pos_y = obs['left_team'][player_num]
        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])

        if ball_distance > 0.03 and obs["game_mode"] == 0:
            illegal_action_set.update([Action.ShortPass, Action.LongPass, Action.HighPass,
                                       Action.Shot, Action.Dribble, Action.Slide])

        if obs['ball_owned_team'] == 1:  # opponents owning ball
            illegal_action_set.update([Action.ShortPass, Action.LongPass, Action.HighPass,
                                       Action.Shot, Action.Dribble])

        elif obs['ball_owned_team'] == 0:  # my team owning ball
            illegal_action_set.update([Action.Slide])

        # Dealing with sticky actions
        sticky_actions = obs['sticky_actions']
        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # sprinting
            illegal_action_set.add(Action.ReleaseSprint)

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # dribbling
            illegal_action_set.add(Action.ReleaseDribble)

        if np.sum(sticky_actions[:8]) == 0:
            illegal_action_set.add(Action.ReleaseDirection)

        # if too far, no shot
        ball_x, ball_y, _ = obs['ball']
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            illegal_action_set.add(Action.Shot)
        elif 0.64 <= ball_x <= 1.0 and -0.27 <= ball_y <= 0.27:
            illegal_action_set.update([Action.LongPass, Action.HighPass])

        if obs['game_mode'] == 2 and ball_x < -0.7:  # Our GoalKick
            illegal_action_set.update([Action.Shot, Action.Sprint, Action.ReleaseDirection, Action.ReleaseSprint,
                                       Action.Slide, Action.Dribble, Action.ReleaseDribble])

        elif obs['game_mode'] == 4 and ball_x > 0.9:  # Our CornerKick
            illegal_action_set.update([Action.Shot, Action.Sprint, Action.ReleaseDirection, Action.ReleaseSprint,
                                       Action.Slide, Action.Dribble, Action.ReleaseDribble])

        elif obs['game_mode'] == 6 and ball_x > 0.6:  # Our PenaltyKick
            illegal_action_set.update([Action.ShortPass, Action.LongPass, Action.HighPass,
                                       Action.Sprint, Action.ReleaseDirection, Action.ReleaseSprint,
                                       Action.Slide, Action.Dribble, Action.ReleaseDribble])

        illegal_action_mask = np.zeros(19)
        illegal_action_mask[list(illegal_action_set)] = 1
        return illegal_action_mask

    @staticmethod
    def encode_ball_which_zone(obs):
        ball_x, ball_y, _ = obs["ball"]
        middle_x, penalty_x, end_x = 0.2, 0.64, 1.0
        penalty_y, end_y = 0.27, 0.42
        if -end_x <= ball_x < -penalty_x and -penalty_y < ball_y < penalty_y:
            return BallZone.PenaltyZone
        elif -end_x <= ball_x < -middle_x and -end_y < ball_y < end_y:
            return BallZone.BackZone
        elif -middle_x <= ball_x <= middle_x and -end_y < ball_y < end_y:
            return BallZone.MidZone
        elif penalty_x < ball_x <= end_x and -penalty_y < ball_y < penalty_y:
            return BallZone.ShotZone
        elif middle_x < ball_x <= end_x and -end_y < ball_y < end_y:
            return BallZone.FrontZone
        else:
            return BallZone.OtherZone

    @staticmethod
    def get_ball_and_player_feature(obs, ball_zone, illegal_action_mask):
        player_num = obs['active']

        player_pos_x, player_pos_y = obs['left_team'][player_num]
        player_direction = np.array(obs['left_team_direction'][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs['left_team_roles'][player_num]
        player_role_onehot = np.zeros(10)
        player_role_onehot[player_role] = 1
        player_tired = obs['left_team_tired_factor'][player_num]
        is_dribbling = obs['sticky_actions'][9]
        is_sprinting = obs['sticky_actions'][8]

        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs['ball_direction']
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        if obs['ball_owned_team'] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        if obs['ball_owned_team'] == 0:
            ball_owned_by_us = 1.0
        elif obs['ball_owned_team'] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone_onehot = np.zeros(6)
        ball_which_zone_onehot[ball_zone] = 1

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        player_state = np.concatenate(
            (illegal_action_mask[9:], obs['left_team'][player_num], player_direction * 100, [player_speed * 100],
             player_role_onehot, [ball_far, player_tired, is_dribbling, is_sprinting]))

        ball_state = np.concatenate((obs['ball'],
                                     ball_which_zone_onehot,
                                     [ball_x_relative, ball_y_relative],
                                     obs['ball_direction'] * 20,
                                     [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]))
        return ball_state, player_state

    @staticmethod
    def get_team_feature(obs, num_left_players=11, num_right_players=11):
        # left players 11*2 -> 11*1*2 -> 11*11*2
        obs_left_team = np.broadcast_to(obs['left_team'][:, None, :], (num_left_players, num_right_players, 2))
        # right players 11*2 -> 1*11*2 -> 11*11*2
        obs_right_team = np.broadcast_to(obs['right_team'][None, ...], (num_left_players, num_right_players, 2))

        # ball  11*11*3
        obs_ball = np.ones((num_left_players, num_right_players, 3)) * obs['ball']
        # goal  11*11*2
        left_goal = np.ones((num_left_players, num_right_players, 2)) * np.array([-1, 0])
        right_goal = np.ones((num_left_players, num_right_players, 2)) * np.array([1, 0])
        # side line  11*11*2
        side_line = np.ones((num_left_players, num_right_players, 2)) * np.array([-0.42, 0.42])
        # active  11*11*2
        active = np.ones((num_left_players, num_right_players, 2)) * obs['active']

        # left players - right players  11*11*2
        left_minus_right_player = obs_left_team - obs_right_team

        # left players - ball, left goal, right goal, side line, active
        left_minus_ball = obs_left_team - obs_ball[..., :2]
        left_minus_left_goal = obs_left_team - left_goal
        left_minus_right_goal = obs_left_team - right_goal
        left_minus_side_line = obs_left_team - side_line
        left_minus_active = obs_left_team - active

        # right players - ball, left goal, right goal, side line, active
        right_minus_ball = obs_right_team - obs_ball[..., :2]
        right_minus_left_goal = obs_right_team - left_goal
        right_minus_right_goal = obs_right_team - right_goal
        right_minus_side_line = obs_right_team - side_line
        right_minus_active = obs_right_team - active

        # left players direction
        obs_left_team_direction = np.broadcast_to(obs['left_team_direction'][:, None, :] * 100,
                                                  (num_left_players, num_right_players, 2))

        # right players direction
        obs_right_team_direction = np.broadcast_to(obs['right_team_direction'][None, ...] * 100,
                                                   (num_left_players, num_right_players, 2))

        # ball direction
        obs_ball_direction = np.ones((num_left_players, num_right_players, 3)) * obs['ball_direction'] * 20

        # left players direction - right players direction
        left_minus_right_player_direction = obs_left_team_direction - obs_right_team_direction

        # left players direction - ball direction
        left_minus_ball_direction = obs_left_team_direction - obs_ball_direction[..., :2]

        # right players direction - ball direction
        right_minus_ball_direction = obs_right_team_direction - obs_ball_direction[..., :2]

        # ball rotation
        obs_ball_rotation =  np.ones((num_left_players, num_right_players, 3)) * obs['ball_rotation'] * 20

        # left team tired factor
        left_team_tired_factor = np.broadcast_to(obs["left_team_tired_factor"][:, None, None],
                                                 (num_left_players, num_right_players, 1))

        # right team tired factor
        right_team_tired_factor = np.broadcast_to(obs["right_team_tired_factor"][None, :, None],
                                                  (num_left_players, num_right_players, 1))

        team_feature = np.concatenate([
            obs_left_team,
            obs_left_team_direction,
            left_team_tired_factor,
            obs_right_team,
            obs_right_team_direction,
            right_team_tired_factor,
            active,
            obs_ball,
            obs_ball_direction,
            obs_ball_rotation,
            left_goal,
            right_goal,
            side_line,
            left_minus_right_player,
            left_minus_right_player_direction,
            left_minus_active,
            left_minus_ball,
            left_minus_ball_direction,
            left_minus_left_goal,
            left_minus_right_goal,
            left_minus_side_line,
            right_minus_active,
            right_minus_ball,
            right_minus_ball_direction,
            right_minus_left_goal,
            right_minus_right_goal,
            right_minus_side_line,
        ], axis=-1)
        return np.transpose(team_feature, (2, 0, 1))

    @staticmethod
    def preprocess_obs(obs, action_history, num_left_players=11, num_right_players=11):
        Observation2Feature._preprocess_obs(obs)
        illegal_action_mask = Observation2Feature.create_illegal_action_masks(obs)
        ball_zone = Observation2Feature.encode_ball_which_zone(obs)
        ball_feature, player_feature = Observation2Feature.get_ball_and_player_feature(obs, ball_zone,
                                                                                       illegal_action_mask)
        team_feature = Observation2Feature.get_team_feature(obs, num_left_players, num_right_players)
        return {
            "team_feature": team_feature,
            "ball_feature": ball_feature,
            "player_feature": player_feature,
            "illegal_action_mask": illegal_action_mask,
            "action_history": np.array(action_history)
        }, ball_zone

    @staticmethod
    def reward(pre_left_yellow_card, pre_right_yellow_card, left_yellow_card, right_yellow_card,
               score, ball_zone, pre_ball_owned_team, ball_owned_team):

        left_yellow = np.sum(left_yellow_card) - np.sum(pre_left_yellow_card)
        right_yellow = np.sum(right_yellow_card) - np.sum(pre_right_yellow_card)
        yellow_r = right_yellow - left_yellow
        if ball_owned_team == 1 and pre_ball_owned_team == 0:
            ball_zone_reward = -2
        elif ball_owned_team == 1:
            ball_zone_reward = -1
        else:
            ball_zone_reward = ball_zone_to_reward[ball_zone]

        return 5 * score + yellow_r + 0.003 * ball_zone_reward


class FeatureEnv(gym.Wrapper):
    def __init__(self, reward_type: Literal["checkpoints", "customized"] = "checkpoints"):
        env = gfootball_env.create_environment(env_name="11_vs_11_easy_stochastic",
                                               render=False,
                                               representation="raw",
                                               rewards="scoring,checkpoints"
                                               if reward_type == "checkpoints" else "scoring",
                                               number_of_left_players_agent_controls=1,
                                               number_of_right_players_agent_controls=0)
        super().__init__(env)
        self.reward_type = reward_type
        self.action_history = collections.deque(maxlen=8)

        self.pre_left_yellow_card = None
        self.pre_right_yellow_card = None
        self.pre_ball_owned_team = 0  # 默认自己拥有球权

    def reset(self):
        obs = self.env.reset()
        for _ in range(random.randint(0, 100)):
            obs, reward, done, info = self.env.step([0])

        obs = obs[0]
        self.action_history.extend([0] * 8)
        feature, _ = Observation2Feature.preprocess_obs(obs, self.action_history)[0]
        self.pre_left_yellow_card = obs["left_team_yellow_card"]
        self.pre_right_yellow_card = obs["right_team_yellow_card"]
        if obs["ball_owned_team"] != -1:
            self.pre_ball_owned_team = obs["ball_owned_team"]
        return feature

    def step(self, action) -> Tuple[Any, Dict, bool, bool, Dict]:
        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])

        truncated = False
        if done:
            truncated = True
            done = False

        obs = obs[0]
        feature, ball_zone = Observation2Feature.preprocess_obs(obs, self.action_history)
        if self.reward_type == "checkpoints":
            reward_infos = {"checkpoints": reward, "scoring": info["score_reward"]}
        else:
            left_yellow_card = obs["left_team_yellow_card"]
            right_yellow_card = obs["right_team_yellow_card"]
            ball_owned_team = obs["ball_owned_team"]
            reward_infos = {"checkpoints": Observation2Feature.reward(self.pre_left_yellow_card,
                                                                      self.pre_right_yellow_card,
                                                                      left_yellow_card,
                                                                      right_yellow_card,
                                                                      reward, ball_zone,
                                                                      self.pre_ball_owned_team,
                                                                      ball_owned_team),
                            "scoring": info["score_reward"]}

            self.pre_right_yellow_card = right_yellow_card
            self.pre_left_yellow_card = left_yellow_card
            if ball_owned_team != -1:
                self.pre_ball_owned_team = obs["ball_owned_team"]

        return feature, reward_infos, done, truncated, info


class FeatureEnv4MultiAgent(gym.Wrapper):
    def __init__(self, num_left=3, num_right=0):
        env = gfootball_env.create_environment(env_name="academy_3_vs_1_with_keeper",
                                               render=False,
                                               representation="raw",
                                               rewards="scoring",
                                               number_of_left_players_agent_controls=num_left,
                                               number_of_right_players_agent_controls=num_right)
        super().__init__(env)
        self.action_histories = [collections.deque(maxlen=8) for _ in range(num_left+num_right)]
        self.num_left = num_left
        self.num_right = num_right

    def reset(self):
        for action_history in self.action_histories:
            action_history.extend([0] * 8)
        obs = self.env.reset()
        for _ in range(random.randint(0, 10)):
            obs, reward, done, info = self.env.step([0] * (self.num_right + self.num_left))
        obs = [Observation2Feature.preprocess_obs(observation, action_history, 4, 2)
               for observation, action_history in zip(obs, self.action_histories)]
        return utils.batchify(obs, unsqueeze=0)

    def step(self, action) -> Tuple[Any, Dict, bool, bool, Dict]:
        for act, action_history in zip(action, self.action_histories):
            action_history.append(act)
        obs, reward, done, info = self.env.step(action)
        reward_infos = {"scoring": info["score_reward"]}
        truncated = False
        obs = [Observation2Feature.preprocess_obs(observation, action_history, 4, 2)
               for observation, action_history in zip(obs, self.action_histories)]

        return utils.batchify(obs, unsqueeze=0), reward_infos, done, truncated, info

#%%


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

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, float], bool, bool, Dict]:
        opponent_action = self.opponent_agent.predict(utils.to_numpy(self.opponent_obs, unsqueeze=0))["action"][0]

        obs, reward, done, info = self.env.step([action] + [opponent_action])

        obs: np.ndarray
        reward: np.ndarray
        done: bool
        info: dict

        reward_infos = {"checkpoints": reward[0], "scoring": info["score_reward"]}

        self.opponent_obs = utils.get_element_from_batch(obs, 1)
        self.scoring += info["score_reward"]

        truncated = False
        if done:
            info["win"] = int(self.scoring > 0)
            info["opponent_id"] = self.opponent_agent.model_id
            truncated = True
            done = False

        return utils.get_element_from_batch(obs, 0), reward_infos, done, truncated, info


class FootballEnv:
    def __init__(self):
        self.envs = {
            "kaggle": gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                                       render=False,
                                                       representation="raw",
                                                       rewards="scoring",
                                                       number_of_left_players_agent_controls=1,
                                                       number_of_right_players_agent_controls=1),
            "kaggle_easy": gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                                            render=False,
                                                            representation="raw",
                                                            rewards="scoring",
                                                            number_of_left_players_agent_controls=1,
                                                            number_of_right_players_agent_controls=1)
        }


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
