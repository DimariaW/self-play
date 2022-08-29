import torch
import logging
import multiprocessing as mp
from tensorboardX import SummaryWriter
from typing import List, Union, Tuple, Dict

from rl.model import Model, ModelValueLogit
from rl.connection import Receiver, send_with_stop_flag


class Algorithm:
    def __init__(self):
        self.tensor_receiver = None

    def learn(self):
        """
        core function, used to fetch data from tensor_receiver and train
        """
        raise NotImplementedError

    def run(self):
        """
        算法在进程中运行需要实现此接口
        """
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights, index):
        raise NotImplementedError

    @staticmethod
    def optimize(optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @staticmethod
    def gradient_clip_and_optimize(optimizer, loss, parameters, max_norm=40.0):
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        logging.debug(f"gradient norm is : {norm.item()}")
        optimizer.step()

    @staticmethod
    def update_q_net(q_estimate, q_target, loss_func, optimizer):
        loss = loss_func(q_estimate, q_target)
        Algorithm.optimize(optimizer, loss)
        return loss.item()

    @staticmethod
    def soft_update(target_model: Model, model: Model, tau=0.005):
        model.sync_weights_to(target_model, 1 - tau)

    @staticmethod
    @torch.no_grad()
    def gae(value: torch.Tensor, reward: torch.Tensor,
            done: torch.Tensor, bootstrap_mask: torch.Tensor,
            gamma: float, lbd: float):  # tensor shape: B*T
        """
        bootstrap_mask = 0, 表示此value仅仅被用于bootstrap, 对此value的value估计保持不变并且adv为0, 且done必须等于True.
        """
        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], value[:, -1:]], dim=-1) - value
        td_error = td_error * bootstrap_mask

        advantage = []
        next_adv = 0

        for i in range(value.shape[1] - 1, -1, -1):
            curr_td_error = td_error[:, i]
            curr_done = done[:, i]
            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * next_adv)
            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=-1)
        return advantage, advantage + value

    @staticmethod
    @torch.no_grad()
    def gae_v2(value: torch.Tensor, reward: torch.Tensor,
               done: torch.Tensor, bootstrap_mask: torch.Tensor,
               gamma: float, lbd: float):  # tensor shape: B*T
        """
        the other version of gae, the speed is slower than the first one, see tests files.
        """
        advantage = []
        next_adv = 0
        next_value = 0

        for i in range(value.shape[1] - 1, -1, -1):
            curr_reward = reward[:, i]
            curr_value = value[:, i]
            curr_done = done[:, i]
            curr_bootstrap_mask = bootstrap_mask[:, i]

            curr_td = curr_reward + (1. - curr_done) * gamma * next_value - curr_value
            curr_td = curr_td * curr_bootstrap_mask
            advantage.insert(0, curr_td + (1. - curr_done) * gamma * lbd * next_adv)

            next_adv = advantage[0]
            next_value = curr_value

        advantage = torch.stack(advantage, dim=-1)  # shape(B, T)
        return advantage, advantage + value

    @staticmethod
    @torch.no_grad()
    def gae_v3(value: torch.Tensor, reward: torch.Tensor,
               done: torch.Tensor, bootstrap_mask: torch.Tensor,
               gamma: float, lbd: float):
        """
        the third version of gae, shed light to the computation of upgo-lambda
        """
        td_value = []
        next_value = 0
        next_td_value = 0

        for i in range(value.shape[1] - 1, -1, -1):
            curr_reward = reward[:, i]
            curr_done = done[:, i]
            curr_bootstrap_mask = bootstrap_mask[:, i]
            curr_value = value[:, i]

            curr_td_value = curr_reward + \
                gamma * (1. - curr_done) * ((1 - lbd) * next_value + lbd * next_td_value)
            curr_td_value = curr_td_value * curr_bootstrap_mask + curr_value * (1. - curr_bootstrap_mask)
            td_value.insert(0, curr_td_value)

            next_value = curr_value
            next_td_value = td_value[0]

        td_value = torch.stack(td_value, dim=-1)

        return td_value - value, td_value

    @staticmethod
    @torch.no_grad()
    def vtrace(value: torch.Tensor, reward: torch.Tensor,
               done: torch.Tensor, bootstrap_mask: torch.Tensor,
               gamma: float, lbd: float, rho, c):  # tensor(B, T)

        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], value[:, -1:]], dim=-1) - value
        td_error = rho * bootstrap_mask * td_error  # bootstrap 的td_error为0

        advantage = []
        next_adv = 0

        for i in range(value.shape[1] - 1, -1, -1):
            curr_td_error = td_error[:, i]
            curr_done = done[:, i]
            curr_c = c[:, i]

            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * curr_c * next_adv)
            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=-1)
        vtrace_value = advantage + value

        advantage = reward + \
            gamma * (1. - done) * torch.concat([vtrace_value[:, 1:], vtrace_value[:, -1:]], dim=-1) - value
        advantage = advantage * bootstrap_mask

        return advantage, vtrace_value

    @staticmethod
    @torch.no_grad()
    def vtrace_multi_action_head(value: torch.Tensor, reward: torch.Tensor,
                                 done: torch.Tensor, bootstrap_mask:torch.Tensor,
                                 gamma: float, lbd: float,
                                 rho: torch.Tensor, c: torch.Tensor):  # rho, c shape (B, T, N)
        value =  value.unsqueeze(-1)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        bootstrap_mask = bootstrap_mask.unsqueeze(-1)

        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], value[:, -1:]], dim=1) - value
        td_error = rho * bootstrap_mask * td_error  # B*T*N

        advantage = []
        next_adv = 0

        for i in range(value.shape[1] - 1, -1, -1):
            curr_td_error = td_error[:, i]  # B*N
            curr_done = done[:, i]  # B*1
            curr_c = c[:, i]  # B*N
            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * curr_c * next_adv) # B*N
            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=1)  # B*T*N
        vtrace_value = advantage + value

        advantage = reward + \
                    gamma * (1. - done) * torch.concat([vtrace_value[:, 1:], vtrace_value[:, -1:]], dim=1) - value
        advantage = advantage * bootstrap_mask

        return advantage, vtrace_value

    @staticmethod
    @torch.no_grad()
    def upgo(value: torch.Tensor, reward: torch.Tensor,
             done: torch.Tensor, bootstrap_mask: torch.Tensor,
             gamma: float, lbd: float):

        target_value = []
        next_value = torch.tensor(0, device=value.device)
        next_target = torch.tensor(0, device=value.device)
        for i in range(value.shape[1] - 1, -1, -1):
            curr_reward = reward[:, i]
            curr_done = done[:, i]
            curr_bootstrap_mask = bootstrap_mask[:, i]
            curr_value = value[:, i]

            curr_target = curr_reward + gamma * (1. - curr_done) *\
                torch.max(next_value, (1 - lbd) * next_value + lbd * next_target)
            curr_target = curr_target * curr_bootstrap_mask + curr_value * (1. - curr_bootstrap_mask)
            target_value.insert(0, curr_target)

            next_value = curr_value
            next_target = target_value[0]

        target_value = torch.stack(target_value, dim=-1)

        return target_value - value, target_value


class ActorCriticBase(Algorithm):
    def __init__(self,
                 model: ModelValueLogit,
                 queue_senders: List[mp.Queue],
                 tensor_receiver: Receiver,
                 lr: float = 2e-3, gamma: float = 0.99, lbd: float = 0.98, vf: float = 0.5, ef: float = 1e-3,
                 tensorboard_dir: str = None):
        super().__init__()
        self.model = model
        # owing to some bug in pytorch 1.12, this statement is not correct, do not need it.
        # self.model.share_memory()
        self.index = torch.tensor(0)
        self.index.share_memory_()  # share model_index
        self.tensor_receiver = tensor_receiver
        self.queue_senders = queue_senders  # used to send model weights (np.ndarray)
        # 2. algorithm hyper-parameters
        self.lr = lr
        self.gamma = gamma
        self.lbd = lbd
        self.vf = vf
        self.ef = ef
        # 3. loss function and optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)  # trick increase eps
        # self.critic_loss_fn = torch.nn.MSELoss()
        self.critic_loss_fn = torch.nn.SmoothL1Loss(reduction="sum")  # trick replace MSE loss
        # 5. tensorboard
        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)

    def set_weights(self, weights, index=None):
        self.model.set_weights(weights)
        if index is not None:
            self.index.data.copy_(torch.tensor(index))

    def get_weights(self):
        return self.model.get_weights()

    def learn(self):
        raise NotImplementedError

    def run(self):
        tag = "learn_info"

        for queue_sender in self.queue_senders:
            send_with_stop_flag(queue_sender, False, (self.index, self.model))

        while True:
            learn_info = self.learn()
            self.index += 1

            index = self.index.item()
            if index % 1 == 0:
                logging.info(f"update num: {index}, {learn_info}")
                if hasattr(self, "sw"):
                    for key, value in learn_info.items():
                        self.sw.add_scalar(f"{tag}/{key}", value, index)


class IMPALA(ActorCriticBase):
    """
    多reward, 单action
    reward 结构应为 {”reward1", "reward2", "reward3"}
    """
    def __init__(self,
                 model: ModelValueLogit,
                 queue_senders: List[mp.Queue],
                 tensor_receiver: Receiver,
                 lr: float = 2e-3, gamma: float = 0.99, lbd: float = 0.98, vf: float = 0.5, ef: float = 1e-3,
                 tensorboard_dir: str = None,
                 vtrace_key: Union[List[str], Tuple[str]] = (),
                 only_critic: Union[List[str], Tuple[str]] = (),
                 upgo_key: Union[List[str], Tuple[str]] = ()
                 ):

        super().__init__(model, queue_senders, tensor_receiver,
                         lr, gamma, lbd, vf, ef, tensorboard_dir)

        assert set(only_critic).issubset(set(vtrace_key))
        self.upgo_key = upgo_key
        self.vtrace_key = vtrace_key
        self.only_critic = only_critic

    def learn(self):
        self.model.train()
        mean_behavior_model_index, batch = self.tensor_receiver.recv()

        obs = batch["observation"]  # shape(B, T)
        behavior_log_prob = batch['behavior_log_prob']  # shape(B, T) or shape(B, T, N)
        action = batch["action"]  # shape(B, T) or shape(B, T, N)
        reward_info: Dict[str, torch.Tensor] = batch["reward_info"]
        done = batch["done"]  # shape(B, T)
        bootstrap_mask = 1. - batch["only_bootstrap"]  # 0表示被mask掉, shape(B, T)

        info = self.model(obs)  # shape: B*T, B*T*N*act_dim | B*T*act_dim
        value_info = info["value_info"]
        action_logits = info["logits"]
        # get behavior log prob
        action_log_prob = torch.log_softmax(action_logits, dim=-1)
        action_log_prob = action_log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # B*T*N | B*T
        log_rho = action_log_prob.detach() - behavior_log_prob
        rho = torch.exp(log_rho)
        # for debugging
        if len(action_log_prob.shape) == 3:
            mean_rho = (torch.sum(rho * bootstrap_mask.unsqueeze(-1)) / torch.sum(bootstrap_mask)).item()
        else:
            mean_rho = (torch.sum(rho * bootstrap_mask) / torch.sum(bootstrap_mask)).item()
        # vtrace clipped rho
        clipped_rho = torch.clamp(rho, 0, 1)  # clip_rho_threshold := 1)  rho shape: B*T*N, B*T
        clipped_c = torch.clamp(rho, 0, 1)  # clip_c_threshold := 1)  c shape: B*T*N, B*T

        actor_losses = {}
        critic_losses = {}
        actor_loss = 0
        critic_loss = 0
        action_head_mask = 1. - obs["illegal_action_head"] \
            if isinstance(obs, dict) and "illegal_action_head" in obs \
            else 1.

        for key in self.vtrace_key:
            value = value_info[key]
            value_nograd = value.detach()
            reward = reward_info[key]

            gae_adv, gae_value = self.gae(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)
            critic_loss_local = self.critic_loss_fn(value, gae_value)
            critic_loss += critic_loss_local
            critic_losses[key + "_critic_loss"] = critic_loss_local.item()
            # vtrace_adv, vtrace_value = self.vtrace(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd,
            #                                      clipped_rho, clipped_c)

            # logging.debug(f" {key} adv is {torch.mean(vtrace_adv)}")
            # logging.debug(f" {key} value is {torch.mean(vtrace_value)}")
            # critic_loss_local = self.critic_loss_fn(value, vtrace_value)
            # critic_loss += critic_loss_local
            # critic_losses[key + "_critic_loss"] = critic_loss_local.item()
            if key not in self.only_critic:
                if len(action_log_prob.shape) == 3:  # shape(B, T, N)
                    vtrace_adv, vtrace_value = self.vtrace_multi_action_head(value_nograd, reward, done, bootstrap_mask,
                                                                             self.gamma, self.lbd,
                                                                             clipped_rho, clipped_c)

                else:
                    vtrace_adv, vtrace_value = self.vtrace(value_nograd, reward, done, bootstrap_mask,
                                                           self.gamma, self.lbd, clipped_rho, clipped_c)

                actor_loss_local = torch.sum(-action_head_mask * action_log_prob * clipped_rho * vtrace_adv)
                actor_loss += actor_loss_local
                actor_losses[key + "_actor_loss"] = actor_loss_local.item()

        if self.upgo_key is not None:
            for key in self.upgo_key:
                value = value_info[key]
                value_nograd = value.detach()
                reward = reward_info[key]

                upgo_adv, upgo_value = self.upgo(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)
                # logging.debug(f" upgo_adv is {torch.mean(upgo_adv)}")
                # logging.debug(f" upgo_value is {torch.mean(upgo_value)}")
                if len(action_log_prob.shape) == 3:
                    actor_loss_local = torch.sum(
                        -action_head_mask * action_log_prob * clipped_rho * upgo_adv.unsqueeze(-1))
                else:
                    actor_loss_local = torch.sum(
                        -action_head_mask * action_log_prob * clipped_rho * upgo_adv)
                actor_loss += actor_loss_local
                actor_losses[key+"_upgo"] = actor_loss_local.item()

        entropy = torch.sum(action_head_mask * torch.distributions.Categorical(logits=action_logits).entropy())

        loss = actor_loss + self.vf * critic_loss - self.ef * entropy

        self.gradient_clip_and_optimize(self.optimizer, loss, self.model.parameters(), 40.0)

        train_info = dict(**actor_losses, **critic_losses,
                          entropy=entropy.item(),
                          data_staleness=self.index.item() - mean_behavior_model_index,
                          rho=mean_rho)
        # when tensor in cuda device, we must delete the variable manually !
        del batch
        return train_info


"""
    def learn(self):
        self.model.train()
        _, batch = self.tensor_receiver.recv()

        obs = batch["observation"]  # shape(B, T)
        action = batch["action"]
        reward_infos: Dict[str, torch.Tensor] = batch["reward_infos"]
        done = batch["done"]

        value_infos, logit = self.model(obs)  # shape(B, T), shape(B, T, action_dim)

        entropy = torch.sum(torch.distributions.Categorical(logits=logit).entropy())

        action_log_probs = torch.log_softmax(logit, dim=-1)
        action_log_prob = torch.gather(action_log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)  # shape(B*T)

        actor_losses = {}
        critic_losses = {}
        actor_loss = 0
        critic_loss = 0

        for key in value_infos.keys():
            value = value_infos[key]
            value_nograd = value.detach()
            reward = reward_infos[key]

            adv, value_estimate = self.a2c_v1(value_nograd[:, :-1], reward[:, :-1],
                                              self.gamma, self.lbd, done[:, :-1], value_nograd[:, -1])
            # trick normalize adv mini-batch
            adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-5)

            actor_loss_local = torch.sum(-action_log_prob[:, :-1] * adv)
            critic_loss_local = self.critic_loss_fn(value[:, :-1], value_estimate)

            actor_loss += actor_loss_local
            critic_loss += critic_loss_local

            actor_losses[key+"_actor_loss"] = actor_loss_local.item()
            critic_losses[key+"_critic_loss"] = critic_loss_local.item()

        self.gradient_clip_and_optimize(self.optimizer,
                                        actor_loss + self.vf * critic_loss - self.ef * entropy,
                                        self.model.parameters(),
                                        max_norm=40.0)
        train_infos = {}
        train_infos.update(**actor_losses, **critic_losses, entropy=entropy.item())
        # when tensor in cuda device, we must delete the variable manually !
        del batch

        return train_infos

"""