import torch
import torch.nn as nn
from collections import OrderedDict

from typing import Dict, Tuple, List, Union
#%% base model class


class Model(nn.Module):
    def __init__(self, name: str = "model"):
        super().__init__()
        self.name = name

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sync_weights_to(self, target_model: "Model", decay=0.0):
        """
        target model and current model must in same device
        """
        target_vars = target_model.state_dict()
        for name, var in self.state_dict().items():
            target_vars[name].data.copy_(decay * target_vars[name].data +
                                         (1 - decay) * var.data)

    def get_weights(self):
        weights = self.state_dict()
        for key in weights.keys():
            weights[key] = weights[key].cpu().numpy()
        return weights

    def set_weights(self, weights):
        new_weights = OrderedDict()
        for key in weights.keys():
            new_weights[key] = torch.from_numpy(weights[key])
        self.load_state_dict(new_weights)
#%% a2c, impala, ppo model class, return value_infos and logits


class ModelValueLogit(Model):
    def forward(self, obs: Union[Dict, Tuple, List, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        obs: batched input
        return: value_infos and logits, support multi-head value.
        """
        raise NotImplementedError

    def init_hidden(self, batch_size: int):
        pass

    def get_hidden(self):
        pass



