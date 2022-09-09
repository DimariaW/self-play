import random
import enum
import gym
import numpy as np
import collections
from typing import Tuple, Dict, Any, Optional, Literal
import gfootball.env as gfootball_env
from rl.agent import Agent
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
    def get_game_feature(obs, half_step):

        def multi_scale(x, scale):
            return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

        # Match state
        if obs['steps_left'] > half_step:
            steps_left_half = obs['steps_left'] - half_step
        else:
            steps_left_half = obs['steps_left']

        game_features = np.concatenate([
            multi_scale(obs['score'], [1, 3, 5]).ravel(),  # 6
            multi_scale(obs['score'][0] - obs['score'][1], [1, 3, 5]),  # 3
            multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),  # 4
            multi_scale(steps_left_half, [10, 100, 1000, 10000]),  # 4
        ]).astype(np.float32)

        mode_index = np.array(obs['game_mode'], dtype=np.int64)

        return {"mode_index": mode_index, 'game': game_features}

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
        game_feature = Observation2Feature.get_game_feature(obs, half_step=1500)
        return {
            "team_feature": team_feature,
            "ball_feature": ball_feature,
            "player_feature": player_feature,
            "illegal_action_mask": illegal_action_mask,
            "action_history": np.array(action_history),
            "game_feature": game_feature
        }

    @staticmethod
    def reward(pre_left_yellow_card, pre_right_yellow_card, left_yellow_card, right_yellow_card,
               score_reward, win_loss,
               ball_zone, pre_ball_owned_team, ball_owned_team):

        left_yellow = np.sum(left_yellow_card) - np.sum(pre_left_yellow_card)
        right_yellow = np.sum(right_yellow_card) - np.sum(pre_right_yellow_card)
        yellow_r = right_yellow - left_yellow

        # if ball_owned_team == 1 and pre_ball_owned_team == 0:  # 丢失球权-0.12奖励
        #   ball_zone_reward = - 40
        #elif ball_owned_team == 1:  # 一直丢失球权 -0.06奖励
         #   ball_zone_reward = - 20
        #elif ball_owned_team == 0:
         #   ball_zone_reward = ball_zone_to_reward[ball_zone]
        #elif ball_owned_team == -1 and pre_ball_owned_team == 1:
         #   ball_zone_reward = 0
        #else:
        ball_zone_reward = ball_zone_to_reward[ball_zone]

        return 5 * (score_reward + win_loss) + yellow_r + 0.003 * ball_zone_reward


KICK_ACTIONS = {
    Action.LongPass: 20,
    Action.HighPass: 28,
    Action.ShortPass: 36,
    Action.Shot: 44,
}


class TamakEriFever:
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

    @staticmethod
    def preprocess_obs(obs: dict, action_history, half_step=1500):
        """
        ・left players (x)
        ・left players (y)
        ・right players (x)
        ・right players (y)
        ・ball (x)
        ・ball (y)
        ・left goal (x)
        ・left goal (y)
        ・right goal (x)
        ・right goal (y)
        ・active (x)
        ・active (y)

        ・left players (x) - right players (x)
        ・left players (y) - right players (y)
        ・left players (x) - ball (x)
        ・left players (y) - ball (y)
        ・left players (x) - goal (x)
        ・left players (y) - goal (y)
        ・left players (x) - active (x)
        ・left players (y) - active (y)

        ・left players direction (x)
        ・left players direction (y)
        ・right players direction (x)
        ・right players direction (y)
        ・left players direction (x) - right players direction (x)
        ・left players direction (y) - right players direction (y)
        """
        TamakEriFever._preprocess_obs(obs)
        illegal_action_mask = TamakEriFever._illegal_actions_mask(obs)
        # left players
        obs_left_team = obs['left_team']
        left_player_x = np.repeat(obs_left_team[:, 0][..., None], 11, axis=1)
        left_player_y = np.repeat(obs_left_team[:, 1][..., None], 11, axis=1)

        # right players
        obs_right_team = obs['right_team']
        right_player_x = np.repeat(obs_right_team[:, 0][..., None], 11, axis=1).transpose((1, 0))
        right_player_y = np.repeat(obs_right_team[:, 1][..., None], 11, axis=1).transpose((1, 0))

        # ball
        obs_ball = obs['ball']
        ball_x = np.ones((11, 11)) * obs_ball[0]
        ball_y = np.ones((11, 11)) * obs_ball[1]
        ball_z = np.ones((11, 11)) * obs_ball[2]

        # goal
        left_goal, right_goal = [-1, 0], [1, 0]
        left_goal_x = np.ones((11, 11)) * left_goal[0]
        left_goal_y = np.ones((11, 11)) * left_goal[1]
        right_goal_x = np.ones((11, 11)) * right_goal[0]
        right_goal_y = np.ones((11, 11)) * right_goal[1]

        # side line
        side_line_y = [-.42, .42]
        side_line_y_top = np.ones((11, 11)) * side_line_y[0]
        side_line_y_bottom = np.ones((11, 11)) * side_line_y[1]

        # active
        active = obs['active']
        active_player_x = np.repeat(obs_left_team[active][0][..., None, None], 11, axis=1).repeat(11, axis=0)
        active_player_y = np.repeat(obs_left_team[active][1][..., None, None], 11, axis=1).repeat(11, axis=0)

        # left players - right players
        left_minus_right_player_x = obs_left_team[:, 0][..., None] - obs_right_team[:, 0]
        left_minus_right_player_y = obs_left_team[:, 1][..., None] - obs_right_team[:, 1]

        # left players - ball
        left_minus_ball_x = (obs_left_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1)
        left_minus_ball_y = (obs_left_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1)

        # left players - right goal
        left_minus_right_goal_x = (obs_left_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1)
        left_minus_right_goal_y = (obs_left_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1)

        # left players - left goal
        left_minus_left_goal_x = (obs_left_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1)
        left_minus_left_goal_y = (obs_left_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1)

        # right players - right goal
        right_minus_right_goal_x = (obs_right_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1).transpose(
            (1, 0))
        right_minus_right_goal_y = (obs_right_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1).transpose(
            (1, 0))

        # right players - left goal
        right_minus_left_goal_x = (obs_right_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1).transpose(1, 0)
        right_minus_left_goal_y = (obs_right_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1).transpose(1, 0)

        # left players (x) - active
        left_minus_active_x = (obs_left_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1)
        left_minus_active_y = (obs_left_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1)

        # right player - ball
        right_minus_ball_x = (obs_right_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1).transpose(1, 0)
        right_minus_ball_y = (obs_right_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1).transpose(1, 0)

        # right player - active
        right_minus_active_x = (obs_right_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11,
                                                                                                   axis=1).transpose(
            1, 0)
        right_minus_active_y = (obs_right_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11,
                                                                                                   axis=1).transpose(
            1, 0)

        # left player - side line
        left_minus_side_top = np.abs(obs_left_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1)
        left_minus_side_bottom = np.abs(obs_left_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1)

        # right player - side line
        right_minus_side_top = np.abs(obs_right_team[:, 1][..., None] - side_line_y[0]).repeat(11,
                                                                                               axis=1).transpose(1,
                                                                                                                 0)
        right_minus_side_bottom = np.abs(obs_right_team[:, 1][..., None] - side_line_y[1]).repeat(11,
                                                                                                  axis=1).transpose(
            1, 0)

        # left players direction
        obs_left_team_direction = obs['left_team_direction']
        left_player_direction_x = np.repeat(obs_left_team_direction[:, 0][..., None], 11, axis=1)
        left_player_direction_y = np.repeat(obs_left_team_direction[:, 1][..., None], 11, axis=1)

        # right players direction
        obs_right_team_direction = obs['right_team_direction']
        right_player_direction_x = np.repeat(obs_right_team_direction[:, 0][..., None], 11, axis=1).transpose((1, 0))
        right_player_direction_y = np.repeat(obs_right_team_direction[:, 1][..., None], 11, axis=1).transpose((1, 0))

        # ball direction
        obs_ball_direction = obs['ball_direction']
        ball_direction_x = np.ones((11, 11)) * obs_ball_direction[0]
        ball_direction_y = np.ones((11, 11)) * obs_ball_direction[1]
        ball_direction_z = np.ones((11, 11)) * obs_ball_direction[2]

        # left players direction - right players direction
        left_minus_right_player_direction_x = obs_left_team_direction[:, 0][..., None] - obs_right_team_direction[:,
                                                                                         0]
        left_minus_right_player_direction_y = obs_left_team_direction[:, 1][..., None] - obs_right_team_direction[:,
                                                                                         1]

        # left players direction - ball direction
        left_minus_ball_direction_x = (obs_left_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11,
                                                                                                                axis=1)
        left_minus_ball_direction_y = (obs_left_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11,
                                                                                                                axis=1)

        # right players direction - ball direction
        right_minus_ball_direction_x = (obs_right_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(
            11, axis=1).transpose(1, 0)
        right_minus_ball_direction_y = (obs_right_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(
            11, axis=1).transpose(1, 0)

        # ball rotation
        obs_ball_rotation = obs['ball_rotation']
        ball_rotation_x = np.ones((11, 11)) * obs_ball_rotation[0]
        ball_rotation_y = np.ones((11, 11)) * obs_ball_rotation[1]
        ball_rotation_z = np.ones((11, 11)) * obs_ball_rotation[2]

        cnn_feature = np.stack([
            left_player_x,
            left_player_y,
            right_player_x,
            right_player_y,
            ball_x,
            ball_y,
            ball_z,
            left_goal_x,
            left_goal_y,
            right_goal_x,
            right_goal_y,
            side_line_y_top,
            side_line_y_bottom,
            active_player_x,
            active_player_y,
            left_minus_right_player_x,
            left_minus_right_player_y,
            left_minus_right_goal_x,
            left_minus_right_goal_y,
            left_minus_left_goal_x,
            left_minus_left_goal_y,
            right_minus_right_goal_x,
            right_minus_right_goal_y,
            right_minus_left_goal_x,
            right_minus_left_goal_y,
            left_minus_side_top,
            left_minus_side_bottom,
            right_minus_side_top,
            right_minus_side_bottom,
            right_minus_ball_x,
            right_minus_ball_y,
            right_minus_active_x,
            right_minus_active_y,
            left_minus_ball_x,
            left_minus_ball_y,
            left_minus_active_x,
            left_minus_active_y,
            ball_direction_x,
            ball_direction_y,
            ball_direction_z,
            left_minus_ball_direction_x,
            left_minus_ball_direction_y,
            right_minus_ball_direction_x,
            right_minus_ball_direction_y,
            left_player_direction_x,
            left_player_direction_y,
            right_player_direction_x,
            right_player_direction_y,
            left_minus_right_player_direction_x,
            left_minus_right_player_direction_y,
            ball_rotation_x,
            ball_rotation_y,
            ball_rotation_z,
        ], axis=0)

        # ball
        BALL_OWEND_1HOT = {-1: [0, 0], 0: [1, 0], 1: [0, 1]}
        ball_owned_team_ = obs['ball_owned_team']
        ball_owned_team = BALL_OWEND_1HOT[ball_owned_team_]  # {-1, 0, 1} None, self, opponent
        PLAYER_1HOT = np.concatenate([np.eye(11), np.zeros((1, 11))])
        ball_owned_player_ = PLAYER_1HOT[obs['ball_owned_player']]  # {-1, N-1}
        if ball_owned_team_ == -1:
            my_ball_owned_player = PLAYER_1HOT[-1]
            op_ball_owned_player = PLAYER_1HOT[-1]
        elif ball_owned_team_ == 0:
            my_ball_owned_player = ball_owned_player_
            op_ball_owned_player = PLAYER_1HOT[-1]
        else:
            my_ball_owned_player = PLAYER_1HOT[-1]
            op_ball_owned_player = ball_owned_player_

        ball_features = np.concatenate([
            obs['ball'],
            obs['ball_direction'],
            obs['ball_rotation']
        ]).astype(np.float32)

        # self team
        left_team_features = np.concatenate([
            [[1] for _ in obs['left_team']],  # left team flag
            obs['left_team'],  # position
            obs['left_team_direction'],
            [[v] for v in obs['left_team_tired_factor']],
            [[v] for v in obs['left_team_yellow_card']],
            [[v] for v in obs['left_team_active']],
            my_ball_owned_player[..., np.newaxis]
        ], axis=1).astype(np.float32)

        left_team_indice = np.arange(0, 11, dtype=np.int32)

        # opponent team
        right_team_features = np.concatenate([
            [[0] for _ in obs['right_team']],  # right team flag
            obs['right_team'],  # position
            obs['right_team_direction'],
            [[v] for v in obs['right_team_tired_factor']],
            [[v] for v in obs['right_team_yellow_card']],
            [[v] for v in obs['right_team_active']],
            op_ball_owned_player[..., np.newaxis]
        ], axis=1).astype(np.float32)

        right_team_indice = np.arange(0, 11, dtype=np.int32)

        # distance information
        def get_distance(xy1, xy2):
            return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

        def get_line_distance(x1, x2):
            return np.abs(x1 - x2)

        def multi_scale(x, scale):
            return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

        both_team = np.concatenate([obs['left_team'], obs['right_team']], axis=0)
        ball = np.array([obs['ball'][:2]], dtype=np.float32)
        goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
        goal_line_x = np.array([-1, 1], dtype=np.float32)
        side_line_y = np.array([-.42, .42], dtype=np.float32)

        # ball <-> goal, goal line, side line distance
        b2g_distance = get_distance(ball, goal)
        b2gl_distance = get_line_distance(ball[0][0], goal_line_x)
        b2sl_distance = get_line_distance(ball[0][1], side_line_y)
        b2o_distance = np.concatenate([
            b2g_distance, b2gl_distance, b2sl_distance
        ], axis=-1)

        # player <-> ball, goal, back line, side line distance
        p2b_distance = get_distance(both_team[:, np.newaxis, :], ball[np.newaxis, :, :])
        p2g_distance = get_distance(both_team[:, np.newaxis, :], goal[np.newaxis, :, :])
        p2gl_distance = get_line_distance(both_team[:, :1], goal_line_x[np.newaxis, :])
        p2sl_distance = get_line_distance(both_team[:, 1:], side_line_y[np.newaxis, :])
        p2bo_distance = np.concatenate([
            p2b_distance, p2g_distance, p2gl_distance, p2sl_distance
        ], axis=-1)

        # player <-> player distance
        p2p_distance = get_distance(both_team[:, np.newaxis, :], both_team[np.newaxis, :, :])

        # apply Multiscale to distances
        # def concat_multiscale(x, scale):
        #    return np.concatenate([x[...,np.newaxis], 1 - multi_scale(x, scale)], axis=-1)

        # distance_scales = [.01, .05, .25, 1.25]
        # b2o_distance = 1 - multi_scale(b2o_distance, distance_scales).reshape(-1)
        # p2bo_distance = 1 - multi_scale(p2bo_distance, distance_scales).reshape(len(both_team), -1)
        # p2p_distance = 1 - multi_scale(p2p_distance, distance_scales).reshape(len(both_team), len(both_team), -1)

        # controlled player information
        control_flag_ = np.array(PLAYER_1HOT[obs['active']], dtype=np.float32)
        control_flag = np.concatenate([control_flag_, np.zeros(len(obs['right_team']))])[..., np.newaxis]

        # controlled status information
        DIR = [
            [-1, 0], [-.707, -.707], [0, 1], [.707, -.707],  # L, TL, T, TR
            [1, 0], [.707, .707], [0, -1], [-.707, .707]  # R, BR, B, BL
        ]
        sticky_direction = DIR[obs['sticky_actions'][:8].argmax()] if 1 in obs['sticky_actions'][:8] else [0, 0]
        sticky_flags = obs['sticky_actions'][8:]

        control_features = np.concatenate([
            sticky_direction,
            sticky_flags,
        ]).astype(np.float32)

        # Match state
        if obs['steps_left'] > half_step:
            steps_left_half = obs['steps_left'] - half_step
        else:
            steps_left_half = obs['steps_left']
        match_features = np.concatenate([
            multi_scale(obs['score'], [1, 3]).ravel(),
            multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),
            multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),
            multi_scale(steps_left_half, [10, 100, 1000, 10000]),
            ball_owned_team,
        ]).astype(np.float32)

        mode_index = np.array([obs['game_mode']], dtype=np.int32)

        # Super Mini Map
        # SMM_WIDTH = 96 #// 3
        # SMM_HEIGHT = 72 #// 3
        # SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

        # # Normalized minimap coordinates
        # MINIMAP_NORM_X_MIN = -1.0
        # MINIMAP_NORM_X_MAX = 1.0
        # MINIMAP_NORM_Y_MIN = -1.0 / 2.25
        # MINIMAP_NORM_Y_MAX = 1.0 / 2.25

        # _MARKER_VALUE = 1  # 255

        # def get_smm_layers(config):
        #     return SMM_LAYERS

        # def mark_points(frame, points):
        #     """Draw dots corresponding to 'points'.
        #     Args:
        #       frame: 2-d matrix representing one SMM channel ([y, x])
        #       points: a list of (x, y) coordinates to be marked
        #     """
        #     for p in range(len(points) // 2):
        #         x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
        #                 (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
        #         y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
        #                 (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
        #         x = max(0, min(frame.shape[1] - 1, x))
        #         y = max(0, min(frame.shape[0] - 1, y))
        #         frame[y, x] = _MARKER_VALUE

        # def generate_smm(observation, config=None,
        #                  channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):
        #     """Returns a list of minimap observations given the raw features for each
        #     active player.
        #     Args:
        #       observation: raw features from the environment
        #       config: environment config
        #       channel_dimensions: resolution of SMM to generate
        #     Returns:
        #       (N, H, W, C) - shaped np array representing SMM. N stands for the number of
        #       players we are controlling.
        #     """
        #     frame = np.zeros((len(observation), channel_dimensions[1],
        #                       channel_dimensions[0], len(get_smm_layers(config))),
        #                       dtype=np.uint8)

        #     for o_i, o in enumerate(observation):
        #         for index, layer in enumerate(get_smm_layers(config)):
        #             assert layer in o
        #             if layer == 'active':
        #                 if o[layer] == -1:
        #                     continue
        #                 mark_points(frame[o_i, :, :, index],
        #                             np.array(o['left_team'][o[layer]]).reshape(-1))
        #             else:
        #                 mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
        #     return frame

        # smm = generate_smm([obs]).transpose(3, 1, 2, 0).squeeze(3).astype(np.float32)

        # ACTION_1HOT = np.eye(19)
        # action_history = np.stack([ACTION_1HOT[a] for a in action_history]).astype(np.float32)
        action_history = np.array(action_history, dtype=np.int32)[..., None]

        return {
            "feature": {
            # features
            'ball': ball_features,
            'match': match_features,
            'player': {
                'self': left_team_features,
                'opp': right_team_features
            },
            'control': control_features,
            'player_index': {
                'self': left_team_indice,
                'opp': right_team_indice
            },
            'mode_index': mode_index,
            'control_flag': control_flag,
            # distances
            'distance': {
                'p2p': p2p_distance,
                'p2bo': p2bo_distance,
                'b2o': b2o_distance
            },
            # CNN
            'cnn_feature': cnn_feature,
            # SuperMiniMap
            # 'smm': smm,
            'action_history': action_history
            },
            "illegal_action_mask": illegal_action_mask
        }

    @staticmethod
    def _illegal_actions_mask(obs):
        illegal_action_mask = np.zeros(52)
        illegal_action_mask[19] = 1  # builtin_ai

        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team != 0:  # not owned or free
            illegal_action_mask[int(Action.LongPass)] = 1
            illegal_action_mask[int(Action.HighPass)] = 1
            illegal_action_mask[int(Action.ShortPass)] = 1
            illegal_action_mask[int(Action.Shot)] = 1
            illegal_action_mask[int(Action.Dribble)] = 1
            for d in range(8):
                illegal_action_mask[KICK_ACTIONS[Action.LongPass] + d] = 1
                illegal_action_mask[KICK_ACTIONS[Action.HighPass] + d] = 1
                illegal_action_mask[KICK_ACTIONS[Action.ShortPass] + d] = 1
                illegal_action_mask[KICK_ACTIONS[Action.Shot] + d] = 1
        else:  # owned
            illegal_action_mask[int(Action.Slide)] = 1

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_action_mask[int(Action.ReleaseSprint)] = 1

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_action_mask[int(Action.ReleaseDribble)] = 1

        if 1 not in sticky_actions[:8]:
            illegal_action_mask[int(Action.ReleaseDirection)] = 1

        return illegal_action_mask


def builtin_ai_observation_to_feature(obs, action_history):
    return obs


class FeatureEnv(gym.Wrapper):
    def __init__(self, reward_type: Literal["checkpoints", "customized", "scoring"] = "checkpoints"):
        env = gfootball_env.create_environment(env_name="11_vs_11_kaggle",
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
        for _ in range(random.randint(0, 0)):
            obs, reward, done, info = self.env.step([0])

        obs = obs[0]
        self.action_history.extend([0] * 8)
        feature, _ = Observation2Feature.preprocess_obs(obs, self.action_history)
        self.pre_left_yellow_card = obs["left_team_yellow_card"]
        self.pre_right_yellow_card = obs["right_team_yellow_card"]
        if obs["ball_owned_team"] != -1:
            self.pre_ball_owned_team = obs["ball_owned_team"]
        return feature

    def step(self, action) -> Tuple[Any, Dict, bool, bool, Dict]:
        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])

        truncated = False
        obs = obs[0]
        win_loss = 0
        if done:
            my_score, opponent_score = obs['score']
            if my_score > opponent_score:
                win_loss = 1
            elif my_score < opponent_score:
                win_loss = -1

        feature, ball_zone = Observation2Feature.preprocess_obs(obs, self.action_history)
        if self.reward_type == "checkpoints":
            reward_infos = {"checkpoints": reward, "scoring": info["score_reward"]}
        elif self.reward_type == "scoring":
            reward_infos = {"checkpoints": info["score_reward"], "scoring": info["score_reward"]}
        else:
            left_yellow_card = obs["left_team_yellow_card"]
            right_yellow_card = obs["right_team_yellow_card"]
            ball_owned_team = obs["ball_owned_team"]
            reward_infos = {"checkpoints": Observation2Feature.reward(self.pre_left_yellow_card,
                                                                      self.pre_right_yellow_card,
                                                                      left_yellow_card,
                                                                      right_yellow_card,
                                                                      info["score_reward"],
                                                                      win_loss,
                                                                      ball_zone,
                                                                      self.pre_ball_owned_team,
                                                                      ball_owned_team),
                            "scoring": info["score_reward"]}

            self.pre_right_yellow_card = right_yellow_card
            self.pre_left_yellow_card = left_yellow_card
            if ball_owned_team != -1:
                self.pre_ball_owned_team = obs["ball_owned_team"]

        return feature, reward_infos, done, truncated, info


class OpponentEnv(gym.Wrapper):
    def __init__(self, agents_pool: Dict[str, Agent] = None):
        env = gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                               render=False,
                                               representation="raw",
                                               rewards="scoring,checkpoints",
                                               number_of_left_players_agent_controls=1,
                                               number_of_right_players_agent_controls=1,
                                               other_config_options={"action_set": "v2"})
        super().__init__(env)
        self.agents_pool = agents_pool if agents_pool is not None else dict()
        self.agents_to_process_func = {
            "feature": Observation2Feature.preprocess_obs,
            "tamak": TamakEriFever.preprocess_obs,
            "builtin_ai": builtin_ai_observation_to_feature
        }
        # opponent info
        self.opponent_agent = None
        self.opponent_preprocess_func = None
        self.opponent_action_history = collections.deque(maxlen=8)
        self.opponent_obs = None
        # current info
        self.preprocess_func = None
        self.action_history = collections.deque(maxlen=8)
        self.scoring = 0

    def add_agent(self, name: str, agent: Agent, process_func=builtin_ai_observation_to_feature):
        assert name not in self.opponents_pool
        assert name == agent.model_id[0]
        self.agents_pool[name] = agent
        self.agents_to_process_func["name"] = process_func

    def reset(self, model_id: Tuple[str, Optional[int]],
              opponent_id: Tuple[str, Optional[int]],
              opponent_weights: Optional):
        model_name, _ = model_id
        self.preprocess_func = self.agents_to_process_func[model_name]

        opponent_name, opponent_index = opponent_id
        self.opponent_agent = self.agents_pool[opponent_name]
        self.opponent_agent.set_weights(opponent_weights, opponent_index)
        self.opponent_preprocess_func = self.agents_to_process_func[opponent_name]

        obs = self.env.reset()
        for _ in range(random.randint(0, 50)):
            obs, reward, done, info = self.env.step([0, 0])

        self.action_history.extend([0] * 8)
        feature = self.preprocess_func(obs[0], self.action_history)

        self.opponent_action_history.extend([0]*8)
        self.opponent_obs = self.opponent_preprocess_func(obs[1], self.opponent_action_history)

        self.scoring = 0
        return feature

    def step(self, action: int) -> Tuple[Any, Dict[str, float], bool, bool, Dict]:
        self.action_history.append(action)

        opponent_action = self.opponent_agent.predict(utils.batchify([self.opponent_obs], unsqueeze=0))["action"][0]
        self.opponent_action_history.append(opponent_action)

        obs, reward, done, info = self.env.step([action] + [opponent_action])

        obs: list
        reward: list
        done: bool
        info: dict

        reward_infos = {"checkpoints": reward[0] - reward[1], "scoring": info["score_reward"]}

        feature = self.preprocess_func(obs[0], self.action_history)
        self.opponent_obs = self.opponent_preprocess_func(obs[1], self.opponent_action_history)

        self.scoring += info["score_reward"]

        truncated = False
        if done:
            if self.scoring == 0:
                info["win"] = 0.5
            elif self.scoring > 0:
                info["win"] = 1
            else:
                info["win"] = 0

            info["opponent_id"] = self.opponent_agent.model_id

        return feature, reward_infos, done, truncated, info

#%%


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

"""
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
    def __init__(self, env, opponents_pool: Dict[str, Agent]):
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
"""


if __name__ == "__main__":
    from tests.football.models import feature_model, tamak_model
    from rl.agent import IMPALAAgent
    import tqdm
    import copy
    import torch
    import pickle
    import logging
    utils.set_process_logger()





    env = OpponentEnv(agents_pool={
        "builtin_ai": feature_model.BuiltinAI(name="builtin_ai"),
        "tamak": tamak_model.TamakAgent(model=tamak_model.FootballNet(name="tamak")),
        "feature": IMPALAAgent(model=feature_model.FeatureModel("feature"))
    }
)
    # env.render()
    """
    timeit = tqdm.tqdm()
    model_id = ("builtin_ai", None)  # ("tamak", 1679)
    opponent_id = ("feature", 126)  # ("builtin_ai", None)

    agent = agents_pool[model_id[0]]
    agent.set_weights(models_pool[model_id], model_id[1])
    obs = env.reset(model_id, opponent_id, opponent_weights=models_pool[opponent_id])

    while True:
        timeit.update()
        action = agent.predict(utils.batchify([obs], unsqueeze=0))["action"][0]
        obs, reward_infos, done, truncated, info = env.step(action)
        if done or truncated:
            logging.info(info)
            model_id, opponent_id = opponent_id, model_id
            agent = agents_pool[model_id[0]]
            agent.set_weights(models_pool[model_id], model_id[1])
            obs = env.reset(model_id, opponent_id, opponent_weights=models_pool[opponent_id])
    """