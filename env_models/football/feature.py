import numpy as np
import torch
import torch.nn as nn

import enum
import rl


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
    """
    namespace of preprocess function for model_name = "feature"
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
    def preprocess_obs(obs, action_history, num_left_players=11, num_right_players=11, half_step=1500):
        Observation2Feature._preprocess_obs(obs)
        illegal_action_mask = Observation2Feature.create_illegal_action_masks(obs)
        ball_zone = Observation2Feature.encode_ball_which_zone(obs)
        ball_feature, player_feature = Observation2Feature.get_ball_and_player_feature(obs, ball_zone,
                                                                                       illegal_action_mask)
        team_feature = Observation2Feature.get_team_feature(obs, num_left_players, num_right_players)
        game_feature = Observation2Feature.get_game_feature(obs, half_step=half_step)
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

        # if ball_owned_team == 1 and pre_ball_owned_team == 0:  # ¶ªÊ§ÇòÈ¨-0.12½±Àø
        #   ball_zone_reward = - 40
        #elif ball_owned_team == 1:  # Ò»Ö±¶ªÊ§ÇòÈ¨ -0.06½±Àø
         #   ball_zone_reward = - 20
        #elif ball_owned_team == 0:
         #   ball_zone_reward = ball_zone_to_reward[ball_zone]
        #elif ball_owned_team == -1 and pre_ball_owned_team == 1:
         #   ball_zone_reward = 0
        #else:
        ball_zone_reward = ball_zone_to_reward[ball_zone]

        return 5 * (score_reward + win_loss) + yellow_r + 0.003 * ball_zone_reward


class GameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_embedding = nn.Embedding(7, 32)
        self.game_embedding = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        self.final_embedding = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )

    def forward(self, game_feature):
        mode_ebd = self.mode_embedding(game_feature["mode_index"])
        game_ebd = self.game_embedding(game_feature["game"])
        return self.final_embedding(torch.concat([mode_ebd, game_ebd], dim=-1))


class Conv1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(55, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.AdaptiveAvgPool2d((1, 11))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 128)
        self.layer_norm = nn.LayerNorm(128)

    def forward(self, feature):
        shape = feature.shape
        feature = feature.view(-1, *shape[-3:])
        feature = self.conv1(feature)
        feature = self.pool1(feature)
        feature = self.conv2(feature)
        feature = self.pool2(feature)
        feature = self.flatten(feature)
        feature = self.layer_norm(self.fc(feature))
        feature = feature.view(*shape[:-3], -1)
        return feature


class ActionHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_emd = nn.Embedding(19, 64)
        self.rnn = nn.GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):  # B*T*8
        h = self.action_emd(x)  # B*T*8*64
        self.rnn.flatten_parameters()
        shape = h.shape
        h = h.view(-1, *shape[-2:])
        h, _ = self.rnn(h)
        return h.view(*shape[:-1], -1)


class FeatureModel(rl.ModelValueLogit):
    def __init__(self, name="feature"):
        super().__init__(name)
        self.player_encoder = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )
        self.ball_encoder = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )
        self.team_encoder = Conv1Model()
        self.action_history_encoder = ActionHistoryEncoder()
        self.game_encoder = GameEncoder()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 20)
        )

    def forward(self, obs):
        player_ebd = self.player_encoder(obs["player_feature"])
        ball_ebd = self.ball_encoder(obs["ball_feature"])
        team_ebd = self.team_encoder(obs["team_feature"])
        action_ebd = self.action_history_encoder(obs["action_history"])
        action_ebd = action_ebd[..., 0, :] + action_ebd[..., -1, :]
        game_ebd = self.game_encoder(obs["game_feature"])
        ebd = torch.concat([player_ebd, ball_ebd, team_ebd, action_ebd, game_ebd], dim=-1)
        value_logits = self.head(ebd)
        return {
            "value_info": {"checkpoints": value_logits[..., 0]},
            "logits": value_logits[..., 1:] - 1e12 * obs["illegal_action_mask"]
        }