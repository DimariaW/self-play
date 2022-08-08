import numpy as np
from typing import Dict
import torch
import rl.model as model_base
import rl.utils as utils

torch.set_num_threads(1)


class Agent:
    """
    convert batched np.ndarray to tensor
    return action_info_dict np.ndarray
    """
    def sample(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def get_weights(self):
        pass

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def model_id(self):
        raise NotImplementedError


class IMPALAAgent(Agent):
    def __init__(self, model: model_base.ModelValueLogit, device=torch.device("cpu"), model_index=None):
        # force the model to device
        self.model = model.to(device)
        self.device = device

        self.model_name = model.name
        self.model_index = model_index

    @torch.no_grad()
    def sample(self, state) -> Dict[str, np.ndarray]:
        self.model.eval()
        state = utils.to_tensor(state, device=self.device)
        _, logits = self.model(state)
        action_idx = torch.distributions.Categorical(logits=logits).sample()
        log_prob = torch.log_softmax(logits, dim=-1)
        behavior_log_prob = torch.gather(log_prob, dim=-1, index=action_idx.unsqueeze(-1)).squeeze(-1)
        return {"action": action_idx.cpu().numpy(), "behavior_log_prob": behavior_log_prob.cpu().numpy()}

    @torch.no_grad()
    def predict(self, state) -> Dict[str, np.ndarray]:
        self.model.eval()
        state = utils.to_tensor(state, device=self.device)
        _, logits = self.model(state)
        action_idx = torch.argmax(logits, dim=-1).cpu().numpy()
        return {"action": action_idx}

    def set_weights(self, weights, model_index=None):
        self.model.set_weights(weights)
        self.model_index = model_index

    def get_weights(self):
        return self.model.get_weights()

    @property
    def model_id(self):
        return self.model_name, self.model_index



