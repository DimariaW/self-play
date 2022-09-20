import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Mapping
import torch
import rl.utils as utils
from rl.model import ModelValueLogit
torch.set_num_threads(1)

ObsType = Union[Dict, Tuple, List, np.ndarray]    # iterable or mapping of batched np.ndarray
WeightsType = Optional[Mapping[str, np.ndarray]]  # agent维护权重的参数，可以为None
IndexType = Optional[int]                         # 此权重的index, 可以为None


class Agent:
    """
    base agent class
    """
    def sample(self, observation: ObsType) -> Dict:
        raise NotImplementedError

    def predict(self, observation: ObsType) -> Dict:
        raise NotImplementedError

    def get_weights(self) -> WeightsType:
        pass

    def set_weights(self, weights: WeightsType, index: IndexType):
        raise NotImplementedError

    @property
    def model_id(self) -> Tuple[str, Optional[int]]:
        raise NotImplementedError

    def init_hidden(self, batch_size: int):
        """
        function that should be implemented when model is rnn
        """
        pass
#%%


class A2CAgent(Agent):
    def __init__(self, model: ModelValueLogit, device=torch.device("cpu"), model_index: Optional[int] = None):
        # force the model to device
        self.model = model.to(device)
        self.device = device

        self.model_name = model.name
        self.model_index = model_index

    @torch.no_grad()
    def sample(self, obs: ObsType) -> Dict:
        self.model.eval()
        obs = utils.to_tensor(obs, device=self.device)
        infos = self.model(obs)
        action_idx = torch.distributions.Categorical(logits=infos["logits"]).sample().cpu().numpy()
        hidden = infos.get("hidden", None)
        if hidden is not None:
            hidden = utils.to_numpy(hidden)
            return {"action": utils.to_numpy(action_idx),
                    "hidden": hidden}
        else:
            return {"action": utils.to_numpy(action_idx)}

    @torch.no_grad()
    def predict(self, obs: ObsType) -> Dict:
        self.model.eval()
        obs = utils.to_tensor(obs, device=self.device)
        infos = self.model(obs)
        action_idx = torch.argmax(infos["logits"], dim=-1).cpu().numpy()
        hidden = infos.get("hidden", None)
        if hidden is not None:
            hidden = utils.to_numpy(hidden)
            return {"action": utils.to_numpy(action_idx),
                    "hidden": hidden}
        else:
            return {"action": utils.to_numpy(action_idx)}

    def set_weights(self, weights: WeightsType, index: IndexType):
        self.model.set_weights(weights)
        self.model_index = index

    def get_weights(self) -> WeightsType:
        return self.model.get_weights()

    @property
    def model_id(self) -> Tuple[str, Optional[int]]:
        return self.model_name, self.model_index

    def init_hidden(self, batch_size: int):
        hidden = self.model.init_hidden(batch_size)
        if hidden is not None:
            hidden = utils.to_numpy(hidden)
        return hidden


class IMPALAAgent(A2CAgent):
    @torch.no_grad()
    def sample(self, observation: ObsType) -> Dict:
        self.model.eval()
        state = utils.to_tensor(observation, device=self.device)
        infos = self.model(state)
        action_idx = torch.distributions.Categorical(logits=infos["logits"]).sample()

        log_prob = torch.log_softmax(infos["logits"], dim=-1)
        behavior_log_prob = torch.gather(log_prob, dim=-1, index=action_idx.unsqueeze(-1)).squeeze(-1)
        hidden = infos.get("hidden", None)
        if hidden is not None:
            hidden = utils.to_numpy(hidden)
            return {"action": utils.to_numpy(action_idx),
                    "behavior_log_prob": utils.to_numpy(behavior_log_prob),
                    "hidden": hidden}
        else:
            return {"action": utils.to_numpy(action_idx),
                    "behavior_log_prob": utils.to_numpy(behavior_log_prob)}


class PPOAgent(A2CAgent):
    @torch.no_grad()
    def sample(self, observation: ObsType) -> Dict:
        self.model.eval()
        state = utils.to_tensor(observation, device=self.device)
        infos = self.model(state)
        # get key and values
        value_infos = infos["value_info"]
        logits = infos["logits"]
        hidden = infos.get("hidden", None)

        for key, value in value_infos.items():
            value_infos[key] = value.cpu().numpy()
        action_idx = torch.distributions.Categorical(logits=logits).sample()
        log_prob = torch.log_softmax(logits, dim=-1)
        behavior_log_prob = torch.gather(log_prob, dim=-1, index=action_idx.unsqueeze(-1)).squeeze(-1)
        if hidden is not None:
            hidden = utils.to_numpy(hidden)
            return {"value_info": value_infos,
                    "action": action_idx.cpu().numpy(),
                    "behavior_log_prob": behavior_log_prob.cpu().numpy(),
                    "hidden": hidden}
        else:
            return {"value_info": value_infos,
                    "action": action_idx.cpu().numpy(),
                    "behavior_log_prob": behavior_log_prob.cpu().numpy()}







