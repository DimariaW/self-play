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

    def init_hidden(self):
        """
        function that should be implemented when model is rnn
        """
        pass

    def get_hidden(self):
        """
        function that get the hidden state ( np.ndarray)
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
    def sample(self, observation: ObsType) -> Dict:
        self.model.eval()
        observation = utils.to_tensor(observation, device=self.device)
        values, logits = self.model(observation)
        action_idx = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for key, value in values.items():
            values[key] = value.cpu().numpy()
        return {"action": action_idx, "values": values}

    @torch.no_grad()
    def predict(self, observation: ObsType) -> Dict:
        self.model.eval()
        observation = utils.to_tensor(observation, device=self.device)
        _, logits = self.model(observation)
        action_idx = torch.argmax(logits, dim=-1).cpu().numpy()
        return {"action": action_idx}

    def set_weights(self, weights: WeightsType, index: IndexType):
        self.model.set_weights(weights)
        self.model_index = index

    def get_weights(self) -> WeightsType:
        return self.model.get_weights()

    @property
    def model_id(self) -> Tuple[str, Optional[int]]:
        return self.model_name, self.model_index


class IMPALAAgent(A2CAgent):
    @torch.no_grad()
    def sample(self, observation: ObsType) -> Dict:
        self.model.eval()
        state = utils.to_tensor(observation, device=self.device)
        _, logits = self.model(state)
        action_idx = torch.distributions.Categorical(logits=logits).sample()
        log_prob = torch.log_softmax(logits, dim=-1)
        behavior_log_prob = torch.gather(log_prob, dim=-1, index=action_idx.unsqueeze(-1)).squeeze(-1)
        return {"action": action_idx.cpu().numpy(), "behavior_log_prob": behavior_log_prob.cpu().numpy()}


class PPOAgent(A2CAgent):
    @torch.no_grad()
    def sample(self, observation: ObsType) -> Dict:
        self.model.eval()
        state = utils.to_tensor(observation, device=self.device)
        values, logits = self.model(state)
        for key, value in values.items():
            values[key] = value.cpu().numpy()
        action_idx = torch.distributions.Categorical(logits=logits).sample()
        log_prob = torch.log_softmax(logits, dim=-1)
        behavior_log_prob = torch.gather(log_prob, dim=-1, index=action_idx.unsqueeze(-1)).squeeze(-1)
        return {"values": values,
                "action": action_idx.cpu().numpy(),
                "behavior_log_prob": behavior_log_prob.cpu().numpy()}






