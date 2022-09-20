import gym
from typing import Tuple, Dict, Any
import logging
import torch
import torch.nn as nn
import rl.model as model


class Env(gym.Wrapper):
    def __init__(self, name, reward_threshold):
        env = gym.make(name)
        self.reward_threshold = reward_threshold
        super().__init__(env)
        """ 
        for CartPole-v1, the threshold is 475
        for LunarLander-v2, the threshold is 200
        """

    def reset(self):
        return self.env.reset()

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        truncated = info.get('TimeLimit.truncated', False)
        if done and truncated:
            done = False
        return obs, {"reward": reward/self.reward_threshold}, done, truncated, info


class OpponentEnv(gym.Wrapper):
    def __init__(self, name, reward_threshold):
        env = gym.make(name)
        self.reward_threshold = reward_threshold
        super().__init__(env)
        """ 
        for CartPole-v1, the threshold is 475
        for LunarLander-v2, the threshold is 200
        """
        self.opponent_id = None

    def reset(self, model_id, opponent_id, opponent_weights):
        """
        opponent_env 默认存在一个agents_pool, 通过在reset的时候指定model_id和opponent_id, 告诉env里的opponent是谁
        至于为什么要传入model_id, 因为要在env中做observation的预处理， 不同的agent对应不同的预处理方式
        """
        self.opponent_id = opponent_id
        return self.env.reset()

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        truncated = info.get('TimeLimit.truncated', False)
        if done:
            if truncated:
                done = False
            info["opponent_id"] = self.opponent_id
            info["win"] = 1
        return obs, {"reward": reward/self.reward_threshold}, done, truncated, info


# orthogonal init
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Model(model.ModelValueLogit):
    def __init__(self, obs_dim, num_action, use_orthogonal_init=True):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_action + 1)
        if use_orthogonal_init:
            logging.info("use orthogonal init")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
        self.act_fn = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> Dict:
        h1 = self.act_fn(self.fc1(obs))
        h2 = self.act_fn(self.fc2(h1))
        output = self.fc3(h2)

        return {"value_info": {"reward": output[..., 0]},
                "logits": output[..., 1:]}  # value and logit, value 的最后一维度需要squeeze


class ModelPseudoRNN(model.ModelValueLogit):
    def __init__(self, obs_dim, num_action):
        super().__init__()
        self.rnn = nn.LSTMCell(input_size=obs_dim,
                               hidden_size=128)
        self.fc = nn.Linear(128, num_action+1)

    def forward(self, obs: Dict):
        observation = obs["observation"]  # shape(B, T, state_dim)
        h, c = obs["hidden"]  # shape(B, T, 128)
        shape = observation.shape
        if len(shape) == 3:  # train model
            observation = observation.view(-1, *shape[2:])
            h = h.view(observation.shape[0], -1)
            c = c.view(observation.shape[0], -1)
            h, _ = self.rnn(observation, (h, c))
            h = h.view(*shape[:2], -1)
            output = self.fc(h)
            return {
                "value_info": {"reward": output[..., 0]},
                "logits": output[..., 1:],
                }

        h, c = self.rnn(observation, (h, c))
        output = self.fc(h)
        return {
            "value_info": {"reward": output[..., 0]},
            "logits": output[..., 1:],
            "hidden": (h, c)
            }

    def init_hidden(self, batch_size: int):
        return torch.zeros(batch_size, 128), torch.zeros(batch_size, 128)


class ModelRNN(model.ModelValueLogit):
    def __init__(self, obs_dim, num_action):
        super().__init__()
        self.rnn = nn.LSTMCell(input_size=obs_dim,
                               hidden_size=128)
        self.fc = nn.Linear(128, num_action + 1)

    def forward(self, obs: Dict):
        observation = obs["observation"]  # shape(B, T, state_dim)
        h, c = obs["hidden"]  # shape(B, T, 128)
        shape = observation.shape
        if len(shape) == 3:  # train model
            h_0 = h[:, 0]  # shape(B, 128)
            c_0 = c[:, 0]  # shape(B, 128)
            init_hidden = obs["init_hidden"].unsqueeze(-1)  # shape(B, T, 1)
            o = []
            for i in range(shape[1]):
                h_0 = h_0 * (1. - init_hidden[:, i]) + h[:, i] * init_hidden[:, i]
                c_0 = c_0 * (1. - init_hidden[:, i]) + c[:, i] * init_hidden[:, i]
                h_0, c_0 = self.rnn(observation[:, i], (h_0, c_0))
                o.append(h_0)

            output = self.fc(torch.stack(o, dim=1))
            return {
                "value_info": {"reward": output[..., 0]},
                "logits": output[..., 1:],
                }

        h, c = self.rnn(observation, (h, c))
        output = self.fc(h)
        return {
            "value_info": {"reward": output[..., 0]},
            "logits": output[..., 1:],
            "hidden": (h, c)
            }

    def init_hidden(self, batch_size: int):
        return torch.zeros(batch_size, 128), torch.zeros(batch_size, 128)


def test():
    import rl.utils as utils
    env = Env("LunarLander-v2", 200)
    obs = env.reset()
    model1 = Model(8, 4)
    model2 = ModelPseudoRNN(8, 4)
    model3 = ModelRNN(8, 4)
    model3.set_weights(model2.get_weights())

    obs = utils.to_tensor(obs, unsqueeze=0)
    hidden = model2.init_hidden(1)
    obs_rnn = {"observation": obs, "hidden": hidden, "init_hidden": False}

    output1 = model1(obs)
    output2 = model2(obs_rnn)
    output3 = model3(obs_rnn)

    print("----")


if __name__ == "__main__":
    test()

