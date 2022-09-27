import time

import torch
import logging
import multiprocessing as mp
from tensorboardX import SummaryWriter
from typing import List, Union, Tuple, Dict, Literal

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
        :param value: shape(batch_size, time_length, ...)
        :param reward: shape(batch_size, time_length, ...), the same as value
        :param done: shape(batch_size, time_length)
        :param bootstrap_mask: shape(batch_size, time_length), 0 represent masked
        :param gamma: discount factor
        :param lbd: td-lambda param.

        value, reward have the same shape, the first two dimension is batch_size, time_length.
        done, bootstrap_mask shape is batch_size, time_length
        bootstrap_mask = 0, 表示此value仅仅被用于bootstrap, reward无意义， done为True。 对此value的value估计保持不变并且adv为0。
        at the  last time step, the done must be true.
        """
        batch_size, time_length = value.shape[:2]
        other_shape = value.shape[2:]
        device = value.device

        done = done.view(batch_size, time_length, * ([1] * len(other_shape)))
        bootstrap_mask = bootstrap_mask.view(batch_size, time_length, *([1] * len(other_shape)))

        bootstrap_value = torch.zeros(batch_size, 1, *other_shape, device=device)
        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], bootstrap_value], dim=1) - value
        td_error = td_error * bootstrap_mask

        advantage = []
        next_adv = 0

        for i in range(time_length - 1, -1, -1):
            curr_td_error = td_error[:, i]
            curr_done = done[:, i]
            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * next_adv)
            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=1)
        return advantage, advantage + value

    @staticmethod
    @torch.no_grad()
    def gae_v2(value: torch.Tensor, reward: torch.Tensor,
               done: torch.Tensor, bootstrap_mask: torch.Tensor,
               gamma: float, lbd: float):  # tensor shape: B*T
        """
        the other version of gae, the speed is slower than the first one, see tests files.
        """
        batch_size, time_length = value.shape[:2]
        other_shape = value.shape[2:]
        done = done.view(batch_size, time_length, *([1] * len(other_shape)))
        bootstrap_mask = bootstrap_mask.view(batch_size, time_length, *([1] * len(other_shape)))

        advantage = []
        next_adv = 0
        next_value = 0

        for i in range(time_length - 1, -1, -1):
            curr_reward = reward[:, i]
            curr_value = value[:, i]
            curr_done = done[:, i]
            curr_bootstrap_mask = bootstrap_mask[:, i]

            curr_td = curr_reward + (1. - curr_done) * gamma * next_value - curr_value
            curr_td = curr_td * curr_bootstrap_mask
            advantage.insert(0, curr_td + (1. - curr_done) * gamma * lbd * next_adv)

            next_adv = advantage[0]
            next_value = curr_value

        advantage = torch.stack(advantage, dim=1)  # shape(B, T)
        return advantage, advantage + value

    @staticmethod
    @torch.no_grad()
    def gae_v3(value: torch.Tensor, reward: torch.Tensor,
               done: torch.Tensor, bootstrap_mask: torch.Tensor,
               gamma: float, lbd: float):
        """
        the third version of gae, shed light to the computation of upgo-lambda
        """
        batch_size, time_length = value.shape[:2]
        other_shape = value.shape[2:]
        done = done.view(batch_size, time_length, *([1] * len(other_shape)))
        bootstrap_mask = bootstrap_mask.view(batch_size, time_length, *([1] * len(other_shape)))

        td_value = []
        next_value = 0
        next_td_value = 0

        for i in range(time_length - 1, -1, -1):
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

        td_value = torch.stack(td_value, dim=1)

        return td_value - value, td_value

    @staticmethod
    @torch.no_grad()
    def vtrace(value: torch.Tensor, reward: torch.Tensor,
               done: torch.Tensor, bootstrap_mask: torch.Tensor,
               gamma: float, lbd: float, rho, c):  # tensor(B, T)
        """
        value, reward shape: B, T, ...  # for multi-agent
        done, bootstrap_mask shape: B, T,
        rho, c shape: B, T, ..., ...    # for multi-action head
        最后一个时间维度的done一定为True, bootstrap也为True
        """
        # 1. get shape info
        batch_size, time_step = done.shape
        agents_shape = value.shape[2:]
        actions_shape = rho.shape[2 + len(agents_shape):]

        # 2. expand
        value = value.view(batch_size, time_step, *agents_shape, *([1]*len(actions_shape)))
        reward = reward.view(batch_size, time_step, *agents_shape, *([1] * len(actions_shape)))
        done = done.view(batch_size, time_step, *([1]*(len(agents_shape) + len(actions_shape))))
        bootstrap_mask = bootstrap_mask.view(batch_size, time_step, *([1]*(len(agents_shape) + len(actions_shape))))

        # 3. create bootstrap
        bootstrap_value = torch.zeros(batch_size, 1, *agents_shape, *([1]*len(actions_shape)), device=value.device)

        # 4. cal td_error
        td_error = reward + gamma * (1. - done) * torch.concat([value[:, 1:], bootstrap_value], dim=1) - value
        td_error = rho * bootstrap_mask * td_error  # bootstrap 的td_error为0

        # 5. cal advantage
        advantage = []
        next_adv = 0
        for i in range(time_step-1, -1, -1):
            curr_td_error = td_error[:, i]
            curr_done = done[:, i]
            curr_c = c[:, i]

            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * curr_c * next_adv)
            next_adv = advantage[0]

        advantage = torch.stack(advantage, dim=1)
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

        batch_size, time_length = value.shape[:2]
        other_shape = value.shape[2:]

        done = done.view(batch_size, time_length, *([1] * len(other_shape)))
        bootstrap_mask = bootstrap_mask.view(batch_size, time_length, *([1] * len(other_shape)))

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

        target_value = torch.stack(target_value, dim=1)

        return target_value - value, target_value


class ActorCriticBase(Algorithm):
    def __init__(self,
                 model: ModelValueLogit,
                 queue_senders: List[mp.Queue],
                 tensor_receiver: Receiver,
                 lr: float, gamma: float, lbd: float, vf: float, ef: float,
                 tensorboard_dir: str,
                 sleep_seconds: float):
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
        self.critic_loss_fn = torch.nn.MSELoss(reduction="sum")
        # self.critic_loss_fn = torch.nn.SmoothL1Loss(reduction="sum")  # trick replace MSE loss
        # 4. tensorboard
        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)
        # 5. sleep_seconds
        self.sleep_seconds = sleep_seconds

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

            # to sleep a few seconds
            time.sleep(self.sleep_seconds)


class IMPALA(ActorCriticBase):
    """
    reward_info 结构应为 {”reward1": float, "reward2": float, "reward3": float}
    value_info 结构应为 {”reward1": float, "reward2": float, "reward3": float}

    critic_loss 三种计算方式：
    1. actor端计算gae 得到value_target_info  learner端认为value_target_info为ground truth, 用来和value_info一起计算loss
    2. actor端计算gae 得到value_target_info, learner仅有value_target_info的最后一个时间步作为bootstrap
    3. 用value_info.detach() 计算gae

    vtrace_key: 用value_info 和 reward计算 adv
    upgo_key: 用value_info 和 reward计算adv

    注意样本最后一个时间维度的done一定为True
    """
    def __init__(self,
                 model: ModelValueLogit,
                 queue_senders: List[mp.Queue],
                 tensor_receiver: Receiver,
                 lr: float = 2e-3, gamma: float = 0.99, lbd: float = 0.98, vf: float = 0.5, ef: float = 1e-3,
                 tensorboard_dir: str = None,
                 sleep_seconds: float = None,
                 # multi-reward
                 critic_key: Union[List[str], Tuple[str]] = (),
                 critic_update_method: Literal["behavior", "behavior_bootstrap", "target"] = "target",
                 vtrace_key: Union[List[str], Tuple[str]] = (),
                 upgo_key: Union[List[str], Tuple[str]] = ()
                 ):

        super().__init__(model, queue_senders, tensor_receiver,
                         lr, gamma, lbd, vf, ef, tensorboard_dir, sleep_seconds)

        assert set(critic_key).issuperset(set(vtrace_key))
        assert set(critic_key).issuperset(set(upgo_key))
        self.critic_key = critic_key
        self.critic_update_method = critic_update_method
        self.vtrace_key = vtrace_key
        self.upgo_key = upgo_key

    def learn(self):
        self.model.train()
        mean_behavior_model_index, batch = self.tensor_receiver.recv()

        obs = batch["observation"]  # shape(B, T, ...)
        behavior_log_prob = batch['behavior_log_prob']  # shape(B, T, ...) or shape(B, T, ..., N)
        action = batch["action"]  # shape(B, T, ...) or shape(B, T, ..., N)
        reward_info: Dict[str, torch.Tensor] = batch["reward_info"]
        done = batch["done"]  # shape(B, T)
        bootstrap_mask = 1. - batch["only_bootstrap"]  # 0表示被mask掉, shape(B, T)

        info = self.model(obs)  # shape: B*T, B*T*N*act_dim | B*T*act_dim
        value_info = info["value_info"]
        action_logits = info["logits"]

        # get log prob
        action_log_prob = torch.log_softmax(action_logits, dim=-1)
        action_log_prob = action_log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # B*T*N | B*T
        log_rho = action_log_prob.detach() - behavior_log_prob
        rho = torch.exp(log_rho)

        # for debugging
        bs_mask_unsqueezed = bootstrap_mask.view(*rho.shape[:2], *([1]*len(rho.shape[2:])))
        mean_rho = (torch.sum(rho * bs_mask_unsqueezed) / torch.sum(bs_mask_unsqueezed)).item()

        # vtrace clipped rho
        clipped_rho = torch.clamp(rho, 0, 1)  # clip_rho_threshold := 1)  rho shape: B*T*N, B*T
        clipped_c = torch.clamp(rho, 0, 1)  # clip_c_threshold := 1)  c shape: B*T*N, B*T

        actor_losses = {}
        critic_losses = {}
        actor_loss = 0
        critic_loss = 0

        action_head_mask = 1. - batch["illegal_action_head"] if "illegal_action_head" in batch else 1.

        if self.critic_update_method == "behavior":
            """
            batch 中存在真value_target值
            """
            value_target_info = batch["value_target_info"]
            for key in self.critic_key:
                value = value_info[key]
                value_target = value_target_info[key]
                critic_loss_local = self.critic_loss_fn(value*bootstrap_mask, value_target*bootstrap_mask)
                critic_loss += critic_loss_local
                critic_losses[key + "_critic_loss"] = critic_loss_local.item()

        elif self.critic_update_method == "behavior_bootstrap":
            value_target_info = batch["value_target_info"]
            for key in self.critic_key:
                value = value_info[key]
                value_target = value_target_info[key]
                value_nograd = value.detach()*bootstrap_mask + value_target*(1.-bootstrap_mask)
                reward = reward_info[key]
                gae_adv, gae_value = self.gae(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)
                critic_loss_local = self.critic_loss_fn(value*bootstrap_mask, gae_value*bootstrap_mask)
                critic_loss += critic_loss_local
                critic_losses[key + "_critic_loss"] = critic_loss_local.item()

        elif self.critic_update_method == "target":
            for key in self.critic_key:
                value = value_info[key]
                value_nograd = value.detach()
                reward = reward_info[key]
                gae_adv, gae_value = self.gae(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)
                critic_loss_local = self.critic_loss_fn(value*bootstrap_mask, gae_value*bootstrap_mask)
                critic_loss += critic_loss_local
                critic_losses[key + "_critic_loss"] = critic_loss_local.item()

        for key in self.vtrace_key:
            value_nograd = value_info[key].detach()
            reward = reward_info[key]
            vtrace_adv, vtrace_value = self.vtrace(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd, clipped_rho, clipped_c)

            actor_loss_local = torch.sum(-action_head_mask * action_log_prob * clipped_rho * vtrace_adv)
            actor_loss += actor_loss_local
            actor_losses[key + "_actor_loss"] = actor_loss_local.item()

        for key in self.upgo_key:
            value_nograd = value_info[key].detach()
            reward = reward_info[key]

            upgo_adv, upgo_value = self.upgo(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)

            actor_loss_local = torch.sum(-action_head_mask * action_log_prob * clipped_rho * upgo_adv)
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


class PPO(ActorCriticBase):
    """
    reward_info 结构应为 {”reward1": float, "reward2": float, "reward3": float}
    value_info 结构应为 {”reward1": float, "reward2": float, "reward3": float}


    Critic 三种更新方式：
    1. actor端计算gae 得到value_target_info  learner端认为value_target_info为ground truth, 用来和value_info一起计算loss
    2. actor端计算gae 得到value_target_info, learner仅有value_target_info的最后一个时间步作为bootstrap
    3. 用value_info.detach() 计算gae
    同时加clip,对比和old policy value 的差距，若差的比较多且value_target也差的比较多，截断？

    actor 三种更新方式：
    1. naive: loss = - clamp(rho, 1-epsilon, 1+epsilon)
    2. standard: loss = -min(clamp(rho, 1-epsilon, 1+epsilon)*ADV, rho*ADV)
    3. dual_clip: loss = max(clamp(rho, 0, ceil)*ADV, -loss_standard)

    vtrace_key: 用value_info 和 reward计算 adv
    upgo_key: 用value_info 和 reward计算adv

    注意样本最后一个时间维度的done一定为True
    """
    def __init__(self,
                 model: ModelValueLogit,
                 queue_senders: List[mp.Queue],
                 tensor_receiver: Receiver,
                 lr: float = 2e-3, gamma: float = 0.99, lbd: float = 0.98, vf: float = 0.5, ef: float = 1e-3,
                 tensorboard_dir: str = None,
                 sleep_seconds: float = None,
                 # multi-reward
                 critic_key: Union[List[str], Tuple[str]] = (),
                 critic_update_method: Literal["behavior", "behavior_bootstrap", "target"] = "target",
                 using_critic_update_method_adv=False,
                 actor_key: Union[List[str], Tuple[str]] = (),
                 actor_update_method: Literal["naive", "standard", "dual_clip"] = "standard"
                 ):

        super().__init__(model, queue_senders, tensor_receiver,
                         lr, gamma, lbd, vf, ef, tensorboard_dir, sleep_seconds)

        assert set(critic_key).issuperset(set(actor_key))
        self.critic_key = critic_key
        self.critic_update_method = critic_update_method
        self.using_critic_update_method_adv = using_critic_update_method_adv
        self.actor_key = actor_key
        self.actor_update_method = actor_update_method

    def learn(self):
        self.model.train()
        mean_behavior_model_index, batch = self.tensor_receiver.recv()
        """
        batch key: observation, behavior_log_prob, action, reward_info, done, only_bootstrap
              optional(key): value_target_info(ground truth value target), 
                             adv_info, 
                             value_info(behavior value info), 
        """
        obs = batch["observation"]  # shape(B, T, ...)
        behavior_log_prob = batch['behavior_log_prob']  # shape(B, T, ...) or shape(B, T, ..., N)
        action = batch["action"]  # shape(B, T, ...) or shape(B, T, ..., N)
        reward_info: Dict[str, torch.Tensor] = batch["reward_info"]
        done = batch["done"]  # shape(B, T)
        bootstrap_mask = 1. - batch["only_bootstrap"]  # 0表示被mask掉, shape(B, T)

        info = self.model(obs)
        value_info = info["value_info"]  # shape: B*T or B*T*N
        action_logits = info["logits"]   # shape: value_dim or value_dim + (action_dim, num_actions)

        # get log prob
        action_log_prob = torch.log_softmax(action_logits, dim=-1)
        action_log_prob = action_log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        log_rho = action_log_prob - behavior_log_prob  # shape: value_dim or value_dim + action_dim
        rho = torch.exp(log_rho)
        clipped_rho = torch.clip(rho, 1-0.1, 1+0.1)

        # for debugging
        rho_nograd = rho.detach()
        clipped_ratio = torch.mean((rho_nograd < 1-0.1).type(torch.float32) + (rho_nograd > 1+0.1).type(torch.float32))
        clipped_ratio = clipped_ratio.item()

        actor_losses = {}
        critic_losses = {}
        actor_loss = 0
        critic_loss = 0

        action_head_mask = 1. - batch["illegal_action_head"] if "illegal_action_head" in batch else 1.

        if self.critic_update_method == "behavior":
            """
            batch 中存在真value_target值
            """
            value_target_info = batch["value_target_info"]
            for key in self.critic_key:
                value = value_info[key]
                value_target = value_target_info[key]
                critic_loss_local = self.critic_loss_fn(value*bootstrap_mask, value_target*bootstrap_mask)
                critic_loss += critic_loss_local
                critic_losses[key + "_critic_loss"] = critic_loss_local.item()

        elif self.critic_update_method == "behavior_bootstrap":
            value_target_info = batch["value_target_info"]
            for key in self.critic_key:
                value = value_info[key]
                value_target = value_target_info[key]
                value_nograd = value.detach()*bootstrap_mask + value_target*(1.-bootstrap_mask)
                reward = reward_info[key]
                gae_adv, gae_value = self.gae(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)
                critic_loss_local = self.critic_loss_fn(value*bootstrap_mask, gae_value*bootstrap_mask)
                critic_loss += critic_loss_local
                critic_losses[key + "_critic_loss"] = critic_loss_local.item()
                if self.using_critic_update_method_adv:
                    batch["adv_info"][key] = gae_adv

        elif self.critic_update_method == "target":
            for key in self.critic_key:
                value = value_info[key]
                value_nograd = value.detach()
                reward = reward_info[key]
                gae_adv, gae_value = self.gae(value_nograd, reward, done, bootstrap_mask, self.gamma, self.lbd)
                critic_loss_local = self.critic_loss_fn(value*bootstrap_mask, gae_value*bootstrap_mask)
                critic_loss += critic_loss_local
                critic_losses[key + "_critic_loss"] = critic_loss_local.item()
                if self.using_critic_update_method_adv:
                    batch["adv_info"][key] = gae_adv

        adv_info = batch["adv_info"]
        if self.actor_update_method == "naive":
            for key in self.actor_key:
                adv = adv_info[key].view(*rho.shape[:2], *([1]*len(rho.shape[2:])))
                actor_loss_local = -torch.sum(action_head_mask*clipped_rho*adv)
                actor_loss += actor_loss_local
                actor_losses[key + "_actor_loss"] = actor_loss_local.item()
        elif self.actor_update_method == "standard":
            for key in self.actor_key:
                adv = adv_info[key].view(*rho.shape[:2], *([1] * len(rho.shape[2:])))
                actor_loss_local = - torch.sum(action_head_mask*torch.minimum(rho*adv, clipped_rho*adv))
                actor_loss += actor_loss_local
                actor_losses[key + "_actor_loss"] = actor_loss_local.item()
        elif self.actor_update_method == "dual_clip":
            for key in self.actor_key:
                adv = adv_info[key].view(*rho.shape[:2], *([1] * len(rho.shape[2:])))
                actor_loss_local = - torch.sum(
                    action_head_mask*torch.maximum(torch.minimum(rho*adv, clipped_rho*adv), torch.clip(rho, 0, 3)*adv)
                )
                actor_loss += actor_loss_local
                actor_losses[key + "_actor_loss"] = actor_loss_local.item()

        entropy = torch.sum(action_head_mask * torch.distributions.Categorical(logits=action_logits).entropy())

        loss = actor_loss + self.vf * critic_loss - self.ef * entropy

        self.gradient_clip_and_optimize(self.optimizer, loss, self.model.parameters(), 40.0)

        train_info = dict(**actor_losses, **critic_losses,
                          entropy=entropy.item(),
                          data_staleness=self.index.item() - mean_behavior_model_index,
                          clipped_ratio=clipped_ratio)
        # when tensor in cuda device, we must delete the variable manually !
        del batch
        return train_info
