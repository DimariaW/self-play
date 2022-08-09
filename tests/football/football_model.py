import torch
import torch.nn as nn
import torch.nn.functional as F
import rl

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
    def __init__(self, in_depth, depths=(16, 32, 32)):
        super(ImpalaCNN, self).__init__()

        layers = []
        for out_depth in depths:
            layers.append(ConvSequence(in_depth, out_depth))
            in_depth = out_depth

        self.residual_layers = nn.ModuleList(layers)

        # in_features size should be derived
        self.fc = nn.Linear(in_features=3456, out_features=256)

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
    def __init__(self, obs_shape=(16, 72, 96), act_dim=19, name: str = "cnn"):
        super().__init__(name)
        """
        obs_shape(CHW)
        """
        self.impala_cnn = ImpalaCNN(in_depth=obs_shape[0])

        self.policy_value_fc = nn.Linear(
            in_features=256,
            out_features=act_dim + 1)

    def forward(self, obs):
        shape = obs.shape  # shape(..., C, H, W)
        obs = obs.view(-1, *shape[-3:])
        out = self.impala_cnn(obs)
        out = out.view(*shape[:-3], -1)
        value_and_logit = self.policy_value_fc(out)
        return {"checkpoints": value_and_logit[..., 0], "scoring": None}, value_and_logit[..., 1:]











