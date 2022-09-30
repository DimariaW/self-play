import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl
import rl.agent as agent
import rl.utils as utils
import enum


# for readable
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


class GameMode(enum.IntEnum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6


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

KICK_ACTIONS = {
    Action.LongPass: 20,
    Action.HighPass: 28,
    Action.ShortPass: 36,
    Action.Shot: 44,
}


class TamakEriFever:
    """
    namespace of preprocess function for model_name = "tamak"
    """
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


class MultiHeadAttention(nn.Module):
    # multi head attention for sets
    # https://github.com/akurniawan/pytorch-transformer/blob/master/modules/attention.py
    def __init__(self, in_dim, out_dim, out_heads, relation_dim=0,
                 residual=False, projection=True, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_heads = out_heads
        self.relation_dim = relation_dim
        assert self.out_dim % self.out_heads == 0
        self.query_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim, bias=False)
        self.key_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim, bias=False)
        self.value_layer = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.residual = residual
        self.projection = projection
        if self.projection:
            self.proj_layer = nn.Linear(self.out_dim, self.out_dim)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = nn.LayerNorm(self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        if self.projection:
            nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)

    def forward(self, query, key, relation=None, mask=None, key_mask=None, distance=None):
        """
        Args:
            query (torch.Tensor): [batch, query_len, in_dim]
            key (torch.Tensor): [batch, key_len, in_dim]
            relation (torch.Tensor): [batch, query_len, key_len, relation_dim]
            mask (torch.Tensor): [batch, query_len]
            key_mask (torch.Tensor): [batch, key_len]
        Returns:
            torch.Tensor: [batch, query_len, out_dim]
        """

        query_len = query.size(-2)
        key_len = key.size(-2)
        head_dim = self.out_dim // self.out_heads

        if key_mask is None:
            if torch.equal(query, key):
                key_mask = mask

        if relation is not None:
            relation = relation.view(-1, query_len, key_len, self.relation_dim)

            query_ = query.view(-1, query_len, 1, self.in_dim).repeat(1, 1, key_len, 1)
            query_ = torch.cat([query_, relation], dim=-1)

            key_ = key.view(-1, 1, key_len, self.in_dim).repeat(1, query_len, 1, 1)
            key_ = torch.cat([key_, relation], dim=-1)

            Q = self.query_layer(query_).view(-1, query_len * key_len, self.out_heads, head_dim)
            K = self.key_layer(key_).view(-1, query_len * key_len, self.out_heads, head_dim)

            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, key_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, query_len, key_len, head_dim)

            attention = (Q * K).sum(dim=-1)
        else:
            Q = self.query_layer(query).view(-1, query_len, self.out_heads, head_dim)
            K = self.key_layer(key).view(-1, key_len, self.out_heads, head_dim)

            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, key_len, head_dim)

            attention = torch.bmm(Q, K.transpose(1, 2))

        if distance is not None:
            attention = attention - torch.log1p(distance.repeat(self.out_heads, 1, 1))
        attention = attention * (float(head_dim) ** -0.5)

        if key_mask is not None:
            attention = attention.view(-1, self.out_heads, query_len, key_len)
            attention = attention + ((1 - key_mask) * -1e32).view(-1, 1, 1, key_len)
        attention = F.softmax(attention, dim=-1)
        if mask is not None:
            attention = attention * mask.view(-1, 1, query_len, 1)
            attention = attention.contiguous().view(-1, query_len, key_len)

        V = self.value_layer(key).view(-1, key_len, self.out_heads, head_dim)
        V = V.transpose(1, 2).contiguous().view(-1, key_len, head_dim)

        output = torch.bmm(attention, V).view(-1, self.out_heads, query_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(*query.size()[:-2], query_len, self.out_dim)

        if self.projection:
            output = self.proj_layer(output)

        if self.residual:
            output = output + query

        if self.layer_norm:
            output = self.ln(output)

        if mask is not None:
            output = output * mask.unsqueeze(-1)
        attention = attention.view(*query.size()[:-2], self.out_heads, query_len, key_len).detach()

        return output, attention


class Dense(nn.Module):
    def __init__(self, units0, units1, bnunits=0, bias=True):
        super().__init__()
        if bnunits > 0:
            bias = False
        self.dense = nn.Linear(units0, units1, bias=bias)
        self.bnunits = bnunits
        self.bn = nn.BatchNorm1d(bnunits) if bnunits > 0 else None

    def forward(self, x):
        h = self.dense(x)
        if self.bn is not None:
            size = h.size()
            h = h.view(-1, self.bnunits)
            h = self.bn(h)
            h = h.view(*size)
        return h


class FootballEncoder(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.player_embedding = nn.Embedding(32, 5, padding_idx=0)
        self.mode_embedding = nn.Embedding(8, 3, padding_idx=0)
        self.fc_teammate = nn.Linear(23, filters)
        self.fc_opponent = nn.Linear(23, filters)
        self.fc = nn.Linear(filters + 41, filters)

    def forward(self, x):
        bs = x['mode_index'].size(0)
        # scalar features
        m_emb = self.mode_embedding(x['mode_index']).view(bs, -1)
        ball = x['ball']
        s = torch.cat([ball, x['match'], x['distance']['b2o'].view(bs, -1), m_emb], dim=1)

        # player features
        p_emb_self = self.player_embedding(x['player_index']['self'])
        ball_concat_self = ball.view(bs, 1, -1).repeat(1, x['player']['self'].size(1), 1)
        p_self = torch.cat([x['player']['self'], p_emb_self, ball_concat_self], dim=2)

        p_emb_opp = self.player_embedding(x['player_index']['opp'])
        ball_concat_opp = ball.view(bs, 1, -1).repeat(1, x['player']['opp'].size(1), 1)
        p_opp = torch.cat([x['player']['opp'], p_emb_opp, ball_concat_opp], dim=2)

        # encoding linear layer
        p_self = self.fc_teammate(p_self)
        p_opp = self.fc_opponent(p_opp)

        p = F.relu(torch.cat([p_self, p_opp], dim=1))
        s_concat = s.view(bs, 1, -1).repeat(1, p.size(1), 1)
        p = torch.cat([p, x['distance']['p2bo'].view(bs, p.size(1), -1), s_concat], dim=2)

        h = F.relu(self.fc(p))

        # relation
        rel = None #x['distance']['p2p']
        distance = None #x['distance']['p2p']

        return h, rel, distance


class FootballBlock(nn.Module):
    def __init__(self, filters, heads):
        super().__init__()
        self.attention = MultiHeadAttention(filters, filters, heads, relation_dim=0,
                                            residual=True, projection=True)

    def forward(self, x, rel, distance=None):
        h, _ = self.attention(x, x, relation=rel, distance=distance)
        return h


class FootballControll(nn.Module):
    def __init__(self, filters, final_filters):
        super().__init__()
        self.filters = filters
        self.attention = MultiHeadAttention(filters, filters, 1, residual=False, projection=True)
        # self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)
        self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)

    def forward(self, x, e, control_flag):
        x_controled = (x * control_flag).sum(dim=1, keepdim=True)
        e_controled = (e * control_flag).sum(dim=1, keepdim=True)

        h, _ = self.attention(x_controled, x)

        h = torch.cat([x_controled, e_controled, h], dim=2).view(x.size(0), -1)
        # h = torch.cat([h, cnn_h.view(cnn_h.size(0), -1)], dim=1)
        h = self.fc_control(h)
        return h


class FootballHead(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.head_p = nn.Linear(filters, 19, bias=False)
        self.head_p_special = nn.Linear(filters, 1 + 8 * 4, bias=False)
        self.head_v = nn.Linear(filters, 1, bias=True)
        self.head_r = nn.Linear(filters, 1, bias=False)

    def forward(self, x):
        p = self.head_p(x)
        p2 = self.head_p_special(x)
        v = self.head_v(x)
        r = self.head_r(x)
        return torch.cat([p, p2], -1), v, r


class CNNModel(nn.Module):
    def __init__(self, final_filters):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(53, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.AdaptiveAvgPool2d((1, 11))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 160, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 96, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, final_filters, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(final_filters),
        )
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x['cnn_feature']
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return x


class ActionHistoryEncoder(nn.Module):
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.action_emd = nn.Embedding(19, 8)
        self.rnn = nn.GRU(8, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        h = self.action_emd(x['action_history'])
        h = h.squeeze(dim=2)
        self.rnn.flatten_parameters()
        h, _ = self.rnn(h)
        return h


class FootballNet(rl.ModelValueLogit):
    def __init__(self, name):
        super().__init__(name)
        blocks = 5
        filters = 96
        final_filters = 128

        self.encoder = FootballEncoder(filters)
        self.blocks = nn.ModuleList([FootballBlock(filters, 8) for _ in range(blocks)])
        self.control = FootballControll(filters, final_filters)  # to head

        self.cnn = CNNModel(final_filters)  # to control
        rnn_hidden = 64
        self.rnn = ActionHistoryEncoder(19, rnn_hidden, 2)
        self.head = FootballHead(final_filters + final_filters + rnn_hidden * 2)

    def forward(self, obs):
        illegal_action_mask = obs["illegal_action_mask"]
        x = obs["feature"]
        e, rel, distance = self.encoder(x)
        h = e
        for block in self.blocks:
            h = block(h, rel, distance)
        cnn_h = self.cnn(x)
        # smm_h = self.smm(x)
        # h = self.control(h, e, x['control_flag'], cnn_h, smm_h)
        h = self.control(h, e, x['control_flag'])
        rnn_h = self.rnn(x)

#         p, v, r = self.head(torch.cat([h,
#                                        cnn_h.view(cnn_h.size(0), -1),
#                                        smm_h.view(smm_h.size(0), -1)], axis=-1))

        rnn_h_head_tail = rnn_h[:, 0, :] + rnn_h[:, -1, :]
        rnn_h_plus_stick = torch.cat([rnn_h_head_tail[:, :-4], x['control']], dim=-1)
        p, v, r = self.head(torch.cat([h, cnn_h.view(cnn_h.size(0), -1), rnn_h_plus_stick], dim=-1))

        return {"logits": p - 1e12 * illegal_action_mask}


class TamakAgent(agent.PPOAgent):
    def __init__(self, *args, **kwargs):
        super(TamakAgent, self).__init__(*args, **kwargs)
        self.reserved_action = None

    def predict(self, obs):
        if self.reserved_action is not None:
            action = self.reserved_action
            self.reserved_action = None
            return {"action": action}

        action = super().predict(utils.get_element_from_batch(obs, 0))["action"][0]
        action, reserved_action = self.special_to_actions(action)
        if reserved_action is not None:
            reserved_action = np.array([[reserved_action]])
        self.reserved_action = reserved_action
        return {"action": np.array([[action]])}

    @staticmethod
    def special_to_actions(action):
        if not 0 <= action < 52:
            return [0, None]
        for a, index in KICK_ACTIONS.items():
            if index <= action < index + 8:
                return [a, Action(action - index + 1)]
        return [action, None]
