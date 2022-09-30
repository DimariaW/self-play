
import logging
import torch
import torch.nn as nn
import rl.model as model
from typing import Dict


# orthogonal init
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Model(model.ModelValueLogit):
    def __init__(self, obs_dim, num_action, hidden_dim, action_heads=1, use_orthogonal_init=True):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_action*action_heads + 1)
        if use_orthogonal_init:
            logging.info("use orthogonal init")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
        self.act_fn = nn.ReLU()
        self.action_heads = action_heads

    def forward(self, obs: torch.Tensor) -> Dict:
        h1 = self.act_fn(self.fc1(obs))
        h2 = self.act_fn(self.fc2(h1))
        output = self.fc3(h2)
        shape = output.shape
        return {"value_info": {"reward": output[..., 0]},
                "logits": output[..., 1:] if self.action_heads == 1
                else output[..., 1:].view(*shape[:-1], self.action_heads, -1)}


class ModelPseudoRNN(model.ModelValueLogit):
    def __init__(self, obs_dim, num_action, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.LSTMCell(input_size=hidden_dim,
                               hidden_size=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_action+1)
        self.hidden_dim = hidden_dim

    def forward(self, obs: Dict):
        observation = obs["observation"]  # shape(B, T, state_dim)
        feature = self.fc1(observation)  # shape(B, T, 128)
        h, c = obs["hidden"]  # shape(B, T, 128)
        shape = feature.shape
        if len(shape) == 3:  # train model
            feature = feature.view(-1, *shape[2:])
            h = h.view(feature.shape[0], -1)
            c = c.view(feature.shape[0], -1)
            h, _ = self.rnn(feature, (h, c))
            h = h.view(*shape[:2], -1)
            output = self.fc3(h)
            return {
                "value_info": {"reward": output[..., 0]},
                "logits": output[..., 1:],
                }

        h, c = self.rnn(feature, (h, c))
        output = self.fc3(h)
        return {
            "value_info": {"reward": output[..., 0]},
            "logits": output[..., 1:],
            "hidden": (h, c)
            }

    def init_hidden(self, batch_size: int):
        return torch.zeros(batch_size, 128), torch.zeros(batch_size, 128)


class ModelRNN(model.ModelValueLogit):
    def __init__(self, obs_dim, num_action, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.LSTMCell(input_size=hidden_dim,
                               hidden_size=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_action + 1)
        self.hidden_dim = hidden_dim

    def forward(self, obs: Dict):
        observation = obs["observation"]  # shape(B, T, state_dim)
        feature = self.fc1(observation)  # shape(B, T, 128)
        h, c = obs["hidden"]  # shape(B, T, 128)
        shape = feature.shape
        if len(shape) == 3:  # train model
            h_0 = h[:, 0]  # shape(B, 128)
            c_0 = c[:, 0]  # shape(B, 128)
            init_hidden = obs["init_hidden"].unsqueeze(-1)  # shape(B, T, 1)
            o = []
            for i in range(shape[1]):
                h_0 = h_0 * (1. - init_hidden[:, i]) + h[:, i] * init_hidden[:, i]
                c_0 = c_0 * (1. - init_hidden[:, i]) + c[:, i] * init_hidden[:, i]
                h_0, c_0 = self.rnn(feature[:, i], (h_0, c_0))
                o.append(h_0)

            output = self.fc3(torch.stack(o, dim=1))
            return {
                "value_info": {"reward": output[..., 0]},
                "logits": output[..., 1:],
                }

        h, c = self.rnn(feature, (h, c))
        output = self.fc3(h)
        return {
            "value_info": {"reward": output[..., 0]},
            "logits": output[..., 1:],
            "hidden": (h, c)
            }

    def init_hidden(self, batch_size: int):
        return torch.zeros(batch_size, 128), torch.zeros(batch_size, 128)