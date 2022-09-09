import numpy as np
import torch
import torch.nn as nn
import rl.agent as agent
import rl
import rl.utils as utils


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
#%%


class BuiltinAI(agent.Agent):
    def __init__(self, name):
        self.name = name

    def predict(self, obs):
        return {"action": np.array([19] * utils.get_batch_size(obs))}

    def sample(self, *args, **kwargs):
        pass

    def set_weights(self, weights=None, model_index=None):
        pass

    def get_weights(self):
        pass

    @property
    def model_id(self):
        return self.name, None








