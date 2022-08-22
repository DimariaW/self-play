import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rl.agent as agent
import rl
import rl.utils as utils

torch.set_num_threads(1)


class ResidualBlock(rl.Model):
    def __init__(self, depth):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding='same')

        self.conv2 = nn.Conv2d(
            in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        out = F.relu(x)

        out = self.conv1(out)

        out = F.relu(out)

        out = self.conv2(out)
        return out + x


class ConvSequence(rl.Model):
    def __init__(self, in_depth, out_depth):
        super(ConvSequence, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_depth, out_channels=out_depth, kernel_size=3, stride=1, padding='same')

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_block1 = ResidualBlock(out_depth)
        self.residual_block2 = ResidualBlock(out_depth)

    def forward(self, x):
        out = self.conv1(x)

        out = self.max_pool(out)

        out = self.residual_block1(out)

        out = self.residual_block2(out)

        return out


class ImpalaCNN(rl.Model):
    def __init__(self, in_depth, depths=(16, 32, 32, 32)):
        super(ImpalaCNN, self).__init__()

        layers = []
        for out_depth in depths:
            layers.append(ConvSequence(in_depth, out_depth))
            in_depth = out_depth

        self.residual_layers = nn.ModuleList(layers)

        # in_features size should be derived
        self.fc = nn.Linear(in_features=960 * (4**(4-len(depths))), out_features=256)

        self.flatten = nn.Flatten()

    def forward(self, x):
        out = x / 255.0

        for layer in self.residual_layers:
            out = layer(out)

        out = self.flatten(out)

        out = F.relu(out)
        out = F.relu(self.fc(out))

        return out


class CNNModel(rl.ModelValueLogit):
    def __init__(self, obs_shape=(16, 72, 96), act_dim=19, name: str = "cnn", depths=(16, 32, 32, 32)):
        super().__init__(name)
        """
        obs_shape(CHW)
        """
        self.impala_cnn = ImpalaCNN(in_depth=obs_shape[0], depths=depths)

        self.policy_value_fc = nn.Linear(
            in_features=256,
            out_features=act_dim + 1)

    def forward(self, observation):
        obs = observation["smm"]
        mask = observation["mask"]
        shape = obs.shape  # shape(..., C, H, W)
        obs = obs.view(-1, *shape[-3:])
        out = self.impala_cnn(obs)
        out = out.view(*shape[:-3], -1)
        value_and_logit = self.policy_value_fc(out)
        return {"checkpoints": value_and_logit[..., 0]}, value_and_logit[..., 1:] - 1e12 * mask
#%%


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

        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 20)
        )

    def forward(self, obs):
        player_ebd = self.player_encoder(obs["player_feature"])
        ball_ebd = self.ball_encoder(obs["ball_feature"])
        team_ebd = self.team_encoder(obs["team_feature"])
        action_ebd = self.action_history_encoder(obs["action_history"])
        action_ebd = action_ebd[..., 0, :] + action_ebd[..., -1, :]

        ebd = torch.concat([player_ebd, ball_ebd, team_ebd, action_ebd], dim=-1)
        value_logits = self.head(ebd)
        return {"checkpoints": value_logits[..., 0]}, value_logits[..., 1:] - 1e12 * obs["illegal_action_mask"]
#%%


class BuiltinAI(agent.Agent):
    def _init__(self, name):
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








