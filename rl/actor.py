import bz2
import os
import pickle
import copy
import numpy as np

from rl.agent import Agent
import rl.connection as connection

from typing import Optional, Tuple, List, Dict, Union, Any, Protocol, Literal, Mapping, Callable
import logging
import multiprocessing as mp

from collections import defaultdict

import rl.utils as utils
from tqdm import tqdm


#%%
class Env(Protocol):
    def reset(self) -> Any:
        # return obs
        ...

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        # return next_obs, reward_infos, done, truncated, info
        ...


class OpponentEnv(Protocol):
    def reset(self, model_id: Tuple[str, Optional[int]], opponent_id: Tuple[str, Optional[int]], opponent_weights) -> Any:
        # return obs
        ...

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        # return next_obs, reward_infos, done, truncated, info
        ...


class Actor:
    """
    how to use this class ?
    env = ...
    agent = ...
    actor = Actor(env, agent, num_steps=32)
    actor.reset_env(*args, **kwargs)
    actor.reset_agent(model_weights, model_index)
    for tag, data in actor.sample_generator():
        if tag == "episodes":
            actor.reset_agent(model_weights, model_index)
            do something
        elif tag == "sample_infos":
            actor.reset_env(*args, **kwargs)
            do something

    或者:
    env = ...
    agent = ...
    actor = Actor(env, agent, num_episodes=1)
    for tag, data in actor.predict_generator():
        if tag == "eval_infos":
            actor.reset_env(*args, **kwargs)
            do something
    """
    def __init__(self,
                 env: Union[Env, OpponentEnv],
                 agents_pool: Dict[str, Agent],
                 # for sample
                 num_steps: int = 0,
                 get_full_episodes: bool = False,
                 # for predict
                 num_episodes: int = 0,
                 # for show sample or predict process
                 process_bar: bool = False,
                 postprocess_meta_info: Callable[[Dict], Dict] = None,
                 postprocess_traj: Callable[[List[Dict]], List[Dict]] = None
                 ):
        """
        :param env:   gym api
        :param agents_pool: 接受batched np.ndarray 作为输入, 返回action_info, 其中必须有action字段
        :param num_steps: 采用当前模型采样num_steps步, used for sample
        :param get_full_episodes: 如果get full_episodes为TRUE, 则actor在采用num_steps步之后继续采样直到episode结束
        :param num_episodes: 采样num_episodes个完整的episode, used for predict
        """
        self.env = env

        for name, agent in agents_pool.items():
            """
            check whether the name equal to the agent model_name
            """
            assert name == agent.model_id[0]
        self.agents_pool = agents_pool

        self.num_steps = num_steps
        self.get_full_episodes = get_full_episodes
        self.num_episodes = num_episodes

        self.process_bar = tqdm() if process_bar else None
        self.postprocess_meta_info = postprocess_meta_info
        self.postprocess_traj = postprocess_traj

        self.total_episodes = 0
        self.agent = None
        self.obs = None
        self.done = None
        self.truncated = None
        self.current_episode_info = defaultdict(float)

    def reset_env(self, *args, **kwargs):
        """ after initialization, reset_env must be called after reset_agent
        model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]
        """
        if self.agent is None:
            raise RuntimeError("must call reset_agent at first!")
        obs = self.env.reset(*args, **kwargs)
        hidden = self.agent.init_hidden(batch_size=1)
        if hidden is not None:
            self.obs = {"observation": obs, "hidden": utils.get_element_from_batch(hidden, 0), "init_hidden": True}
        else:
            self.obs = obs
        self.done = False
        self.truncated = False
        self.current_episode_info.clear()

    def reset_agent(self, model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]):
        """model_id = (model_name, model_index)
        assert(self.agent.model_id[0] = model_id[0])
        """
        name, index = model_id
        self.agent = self.agents_pool[name]
        self.agent.set_weights(model_weights, index)

    @staticmethod
    def _update_moment_from_action_info(moment, action_info):
        """
        update moment from action_info except hidden
        """
        for key, value in action_info.items():
            if key != "hidden":
                moment[key] = value

    def _update_obs_from_action_info(self, obs, action_info):
        hidden = action_info.get("hidden", None)
        if hidden is not None:
            self.obs = {"observation": obs, "hidden": hidden, "init_hidden": False}
        else:
            self.obs = obs

    def _update_current_episode_info(self, reward_info):
        self.current_episode_info["steps"] += 1
        for key, value in reward_info.items():
            self.current_episode_info[key] += value

    def _record_update_when_done(self, env_info):
        """
        1. record according to num_episodes, current_episode_infos and ending_info
        """
        self.total_episodes += 1

        rewards_per_agent = {}
        keys_for_delete = []
        for key, value in self.current_episode_info.items():
            if isinstance(value, np.ndarray):
                for index, reward in enumerate(value):
                    rewards_per_agent[f"{key}_{index}"] = reward
                keys_for_delete.append(key)
        for key in keys_for_delete:
            del self.current_episode_info[key]
        self.current_episode_info.update(rewards_per_agent)

        meta_info = env_info
        meta_info.update(model_id=self.agent.model_id)
        self.current_episode_info.update(meta_info=meta_info)
        if self.postprocess_meta_info is not None:
            self.current_episode_info.update(self.postprocess_meta_info(meta_info))

        logging.info(f"num_episodes is : {self.total_episodes}, "
                     f"current episode_infos is : {self.current_episode_info}, "
                     f"ending info is {env_info}")

    def _update_process_bar(self):
        if self.process_bar is not None:
            self.process_bar.update()

    def _sample_a_step(self):
        self._update_process_bar()
        moment = dict()
        moment['observation'] = self.obs
        # sample from batch obs and get hidden and action
        action_info_batched = self.agent.sample(utils.batchify([self.obs], unsqueeze=0))
        # action for env.step, hidden for next obs
        action_info = utils.get_element_from_batch(action_info_batched, 0)
        self._update_moment_from_action_info(moment, action_info)
        # step
        obs, reward_info, self.done, self.truncated, info = self.env.step(moment['action'])
        moment["reward_info"] = reward_info
        moment["done"] = self.done
        moment["only_bootstrap"] = False
        # update obs
        self._update_obs_from_action_info(obs, action_info)
        # update info
        self._update_current_episode_info(reward_info)
        return moment, info

    def _append_bootstrap_moment(self, moment_list):
        """
        need to get real value_infos, but do not call env.step.
        """
        bootstrap_moment = copy.deepcopy(moment_list[-1])
        bootstrap_moment["observation"] = self.obs
        action_info_batched = self.agent.sample(utils.batchify([self.obs], unsqueeze=0))
        # action for env.step, hidden for next obs
        action_info = utils.get_element_from_batch(action_info_batched, 0)
        self._update_moment_from_action_info(bootstrap_moment, action_info)
        bootstrap_moment["done"] = True
        bootstrap_moment["only_bootstrap"] = True
        moment_list.append(bootstrap_moment)

    def _sample_num_steps(self):
        # episode generation
        if self.obs is None:
            raise RuntimeError("need reset env in advance!")

        moment_list = []
        step = 0
        while step < self.num_steps:
            moment, info = self._sample_a_step()
            moment_list.append(moment)
            step += 1
            if self.done or self.truncated:
                if self.truncated:
                    self._append_bootstrap_moment(moment_list)
                    step += 1

                self._record_update_when_done(info)
                yield "sample_infos", self.current_episode_info  # scoring, checkpoints, model_id, opponent_id
                # need reset env outside
                if self.done or self.truncated:
                    raise RuntimeError("need call reset env outside the generator!")

        if step == self.num_steps:
            self._append_bootstrap_moment(moment_list)

        if self.postprocess_traj is not None:
            moment_list = self.postprocess_traj(moment_list)
        yield "episodes", (self.agent.model_id[1], moment_list)

    def _sample_full_episodes(self):
        """
        note: sample_full_episodes not only sample an episode,
        but also may sample many episodes to ensure its length greater than num_steps
        """
        # episode generation
        if self.obs is None:
            raise RuntimeError("need reset env  in advance!")

        moment_list = []
        step = 0
        while True:
            moment, info = self._sample_a_step()
            moment_list.append(moment)
            step += 1

            if self.done or self.truncated:
                if self.truncated:
                    self._append_bootstrap_moment(moment_list)
                    step += 1

                self._record_update_when_done(info)
                yield "sample_infos", self.current_episode_info  # scoring, checkpoints, model_id, opponent_id
                # need reset env outside
                if self.done or self.truncated:
                    raise RuntimeError("need call reset env outside the generator!")

                if step >= self.num_steps:
                    break

        if self.postprocess_traj is not None:
            moment_list = self.postprocess_traj(moment_list)
        yield "episodes", (self.agent.model_id[1], moment_list)

    def sample(self):
        if self.get_full_episodes:
            return self._sample_full_episodes()
        else:
            return self._sample_num_steps()

    def sample_generator(self):
        """
        infinitely sample
        """
        while True:
            sample = self.sample()
            for cmd, data in sample:
                yield cmd, data

    def predict(self):
        if self.obs is None:
            raise RuntimeError("need call reset env advance!")

        num_episodes = 0

        while num_episodes < self.num_episodes:
            self._update_process_bar()
            action_info_batched = self.agent.predict(utils.batchify([self.obs], unsqueeze=0))
            action_info = utils.get_element_from_batch(action_info_batched, 0)
            obs, reward_info, self.done, self.truncated, info = self.env.step(action_info['action'])
            self._update_obs_from_action_info(obs, action_info)
            self._update_current_episode_info(reward_info)

            if self.done or self.truncated:
                num_episodes += 1
                self._record_update_when_done(info)
                yield "eval_infos", self.current_episode_info  # scoring, checkpoints, model_id, opponent_id
                if self.done or self.truncated:
                    raise RuntimeError("need call reset env outside the generator!")

    def predict_generator(self):
        """
        infinitely predict
        """
        while True:
            predict = self.predict()
            for cmd, data in predict:
                yield cmd, data

    @staticmethod
    # TODO: remove in future version.
    def _gae(value: np.ndarray, reward: np.ndarray,
             done: np.ndarray, bootstrap_mask: np.ndarray,
             gamma: float, lbd: float):  # array shape(T*B),  done shape(T), bootstrap_mask shape(T)
        """
        bootstrap_mask = 0, 表示此value仅仅被用于bootstrap, 对此value的value估计保持不变并且adv为0, 且done必须等于True.
        """
        done = done[:, None]  # T*1
        bootstrap_mask = bootstrap_mask[:, None]  # T*1

        td_error = reward + gamma * (1. - done) * np.concatenate([value[1:], value[-1:]], axis=0) - value
        td_error = td_error * bootstrap_mask  # T*B

        advantage = []
        next_adv = 0

        for i in range(value.shape[0] - 1, -1, -1):
            curr_td_error = td_error[i]  # B
            curr_done = done[i]
            advantage.insert(0, curr_td_error + gamma * (1. - curr_done) * lbd * next_adv)  # B
            next_adv = advantage[0]

        advantage = np.stack(advantage, axis=0)
        return advantage, advantage + value

    @staticmethod
    # TODO: remove in future version.
    def get_cal_gae_function(gamma_infos: Union[Dict, float] = 0.99, lbd_infos: Union[Dict, float] = 1):
        def cal_gea_from_episodes(episodes):
            value_infos = utils.batchify([episode["value_info"] for episode in episodes], unsqueeze=0)
            done = utils.batchify([episode["done"] for episode in episodes], unsqueeze=0)
            bootstrap_mask = 1. - utils.batchify([episode["only_bootstrap"] for episode in episodes], unsqueeze=0)

            adv_infos = [{} for _ in range(len(episodes))]
            value_target_infos = [{} for _ in range(len(episodes))]
            for key, value in value_infos.items():
                reward = utils.batchify([episode["reward_info"][key] for episode in episodes], unsqueeze=0)
                gamma = gamma_infos[key] if isinstance(gamma_infos, dict) else gamma_infos
                lbd = lbd_infos[key] if isinstance(lbd_infos, dict) else lbd_infos

                if len(value.shape) == 1:
                    advantage, value = Actor._gae(value[:, None], reward[:, None], done, bootstrap_mask, gamma, lbd)
                    advantage = advantage.squeeze(axis=-1)
                    value = value.squeeze(axis=-1)
                elif len(value.shape) == 2:
                    advantage, value = Actor._gae(value, reward, done, bootstrap_mask, gamma, lbd)
                else:
                    raise RuntimeError(f"invalid value shape: {value.shape}")
                for adv_info, vt_info, adv, vt in zip(adv_infos, value_target_infos, advantage, value):
                    adv_info[key] = adv
                    vt_info[key] = vt

            for adv_info, vt_info, epi in zip(adv_infos, value_target_infos, episodes):
                del epi["value_info"]
                del epi["reward_info"]
                epi["adv_info"] = adv_info
                epi["value_target_info"] = vt_info
            return episodes
        return cal_gea_from_episodes


class PostProcess:
    @staticmethod
    def get_cal_gae_func(gamma_infos: Union[Dict, float] = 0.99, lbd_infos: Union[Dict, float] = 1.):
        if isinstance(gamma_infos, (float, int)):
            gamma = gamma_infos
            gamma_infos = defaultdict(lambda: gamma)
        if isinstance(lbd_infos, (float, int)):
            lbd = lbd_infos
            lbd_infos = defaultdict(lambda: lbd)

        def cal_gae(episodes):
            next_adv = defaultdict(float)
            next_value = defaultdict(float)
            for moment in reversed(episodes):
                curr_reward = moment["reward_info"]
                curr_value = moment["value_info"]
                curr_done = moment["done"]
                curr_bs_mask = 1. - moment["only_bootstrap"]

                curr_td = {}
                curr_adv = {}
                curr_value_target = {}
                for key in curr_value.keys():
                    curr_td[key] = curr_reward[key] + (1. - curr_done) * gamma_infos[key] * next_value[key] - curr_value[key]
                    curr_td[key] = curr_td[key] * curr_bs_mask
                    curr_adv[key] = curr_td[key] + (1. - curr_done) * gamma_infos[key] * lbd_infos[key] * next_adv[key]
                    curr_value_target[key] = curr_adv[key] + curr_value[key]
                moment["adv_info"] = curr_adv
                moment["value_target_info"] = curr_value_target

                next_adv = curr_adv
                next_value = curr_value

            for moment in episodes:
                del moment["value_info"]
                # del moment["reward_info"]

            return episodes
        return cal_gae


class ActorSampler:
    def __init__(self,
                 env: Env,
                 agents_pool: Dict[str, Agent],
                 num_steps: int = 0,
                 get_full_episodes: bool = False,
                 process_bar: bool = False,
                 postprocess_meta_info: Callable[[Dict], Dict] = None,
                 postprocess_traj: Callable[[List[Dict]], List[Dict]] = None):

        self._actor = Actor(env, agents_pool, num_steps, get_full_episodes,
                            process_bar=process_bar,
                            postprocess_meta_info=postprocess_meta_info,
                            postprocess_traj=postprocess_traj)

    def reset_env(self, *args, **kwargs):
        self._actor.reset_env(*args, **kwargs)

    def reset_agent(self, model_id: Tuple[str, int], model_weights: Optional[Mapping[str, np.ndarray]]):
        self._actor.reset_agent(model_id, model_weights)

    def sample(self):
        yield from self._actor.sample()

    def sample_generator(self):
        yield from self._actor.sample_generator()


class ActorEvaluator:
    def __init__(self,
                 env: Env,
                 agents_pool: Dict[str, Agent],
                 num_episodes: int = 0,
                 process_bar: bool = False,
                 postprocess_meta_info: Callable[[Dict], Dict] = None
                 ):
        self._actor = Actor(env, agents_pool, num_episodes=num_episodes,
                            process_bar=process_bar,
                            postprocess_meta_info=postprocess_meta_info)

    def reset_env(self, *args, **kwargs):
        self._actor.reset_env(*args, **kwargs)

    def reset_agent(self, model_id: Tuple[str, Optional[int]], model_weights: Optional[Mapping[str, np.ndarray]]):
        self._actor.reset_agent(model_id, model_weights)

    def predict(self):
        yield from self._actor.predict()

    def predict_generator(self):
        yield from self._actor.predict_generator()


#%%
"""
编程规范: 对每个通信写logging.debug, 格式为send ..., receive ..., 写到基类中。
"""


def send_with_sender_id_and_receive(queue_sender: mp.Queue, queue_receiver: mp.Queue,
                                    sender_id: int, data: Tuple[str, Any],
                                    block: bool = True, timeout: float = None) -> Tuple[str, Any]:
    queue_sender.put((sender_id, data), block=block, timeout=timeout)
    cmd, data = queue_receiver.get()
    return cmd, data


class ActorManagerBase:
    def __init__(self,
                 actor_id: int,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 ):
        self.actor_id = actor_id
        self.queue_gather2actor = queue_gather2actor
        self.queue_actor2gather = queue_actor2gather

        self.use_bz2 = None
        self.compress_step_length = None
        self.self_play = None
        self.model_id = None

    def _request_weights(self, cmd: Literal["model"], data: Literal["latest",
                                                                    "sample_opponent",
                                                                    "eval_model_and_opponent"]):

        if data == "latest" or data == "sample_opponent":
            """1. request weights
            """
            cmd, (model_id, weights) = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                       self.queue_gather2actor,
                                                                       self.actor_id,
                                                                       (cmd, data))
            logging.debug(f"send (cmd: {cmd}, data: {data}), receive (cmd: {cmd}, data: {model_id})")
            """2. decompress?
            """
            if weights is not None and self.use_bz2:
                weights = pickle.loads(bz2.decompress(weights))
            return model_id, weights

        elif data == "eval_model_and_opponent":
            """1. request weights
            """
            cmd, ((model_id, weights), (opponent_id, opponent_weights)) = send_with_sender_id_and_receive(
                self.queue_actor2gather, self.queue_gather2actor, self.actor_id, (cmd, data))
            logging.debug(f"send (cmd: {cmd}, data: {data}), receive (cmd: {cmd}, data: {model_id}, {opponent_id})")
            """2. decompress?
            """
            if weights is not None and self.use_bz2:
                weights = pickle.loads(bz2.decompress(weights))
            if opponent_weights is not None and self.use_bz2:
                opponent_weights = pickle.loads(bz2.decompress(opponent_weights))
            return (model_id, weights), (opponent_id, opponent_weights)

        else:
            raise ValueError(f"data is {data}, which is not in latest, sample_opponent, eval_model_and_opponent")

    def _send_infos(self, cmd: Literal["sample_infos", "eval_infos"], data):
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   sender_id=self.actor_id,
                                                   data=(cmd, data))
        logging.debug(f"send cmd: {cmd}, receive: {msg}")

    def _send_episodes(self, cmd: Literal["episodes"], data):
        if self.use_bz2:
            if self.compress_step_length == "full":
                index, episodes = data
                episodes = bz2.compress(pickle.dumps(episodes))
                data = (index, episodes)
            else:
                index, episodes = data
                episodes = [bz2.compress(pickle.dumps(episodes[i:i+self.compress_step_length]))
                            for i in range(0, len(episodes), self.compress_step_length)]
                data = (index, episodes)
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   self.actor_id,
                                                   (cmd, data))
        logging.debug(f"send cmd: {cmd}, receive: {msg}")

    def run(self):
        if not self.self_play:
            self._run()
        else:
            self._run_self_play()

    def _run(self):
        raise NotImplementedError

    def _run_self_play(self):
        raise NotImplementedError


class ActorSamplerManager(ActorManagerBase):
    def __init__(self,
                 actor_id: int,
                 actor: ActorSampler,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 use_bz2: bool, compress_step_length: Union[Literal["full"], int], self_play: bool):
        super(ActorSamplerManager, self).__init__(actor_id, queue_gather2actor, queue_actor2gather)
        self.actor = actor
        self.use_bz2 = use_bz2
        self.compress_step_length = compress_step_length
        self.self_play = self_play

    def _run(self):
        cmd2func = {
            "episodes": self.send_episodes_and_reset_agent,
            "sample_infos": self.send_sample_infos_and_reset_env,
        }
        self.model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(self.model_id, weights)
        self.actor.reset_env()

        sample_generator = self.actor.sample_generator()
        for cmd, data in sample_generator:
            cmd2func[cmd](cmd, data)

    def _run_self_play(self):
        cmd2func = {
            "episodes": self.send_episodes_and_reset_agent,
            "sample_infos": self.send_sample_infos_and_reset_env,
        }
        self.model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(self.model_id, weights)

        opponent_id, opponent_weights = self._request_weights("model", "sample_opponent")
        self.actor.reset_env(self.model_id, opponent_id, opponent_weights)

        sample_generator = self.actor.sample_generator()
        for cmd, data in sample_generator:
            cmd2func[cmd](cmd, data)

    def send_episodes_and_reset_agent(self, cmd, data):
        self._send_episodes(cmd, data)
        self.model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(self.model_id, weights)

    def send_sample_infos_and_reset_env(self, cmd, data):
        self._send_infos(cmd, data)
        if self.self_play:
            opponent_id, opponent_weights = self._request_weights("model", "sample_opponent")
            self.actor.reset_env(self.model_id, opponent_id, opponent_weights)
        else:
            self.actor.reset_env()


class ActorEvaluatorManager(ActorManagerBase):
    def __init__(self,
                 actor_id: int,
                 actor: ActorEvaluator,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 use_bz2: bool, self_play: bool):
        super(ActorEvaluatorManager, self).__init__(actor_id, queue_gather2actor, queue_actor2gather)
        self.actor = actor
        self.use_bz2 = use_bz2
        self.self_play = self_play

    def _run(self):
        cmd2func = {
            "eval_infos": self.send_eval_infos_and_reset_agent_env
        }
        self.model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(self.model_id, weights)
        self.actor.reset_env()

        predict_generator = self.actor.predict_generator()
        for cmd, data in predict_generator:
            cmd2func[cmd](cmd, data)

    def _run_self_play(self):
        cmd2func = {
            "eval_infos": self.send_eval_infos_and_reset_agent_env
        }

        (self.model_id, weights), (opponent_id, opponent_weights) = \
            self._request_weights("model", "eval_model_and_opponent")
        self.actor.reset_agent(self.model_id, weights)
        self.actor.reset_env(self.model_id, opponent_id, opponent_weights)

        predict_generator = self.actor.predict_generator()
        for cmd, data in predict_generator:
            cmd2func[cmd](cmd, data)

    def send_eval_infos_and_reset_agent_env(self, cmd, data):
        self._send_infos(cmd, data)
        if not self.self_play:
            self.model_id, weights = self._request_weights("model", "latest")
            self.actor.reset_agent(self.model_id, weights)
            self.actor.reset_env()
        else:
            (self.model_id, weights), (opponent_id, opponent_weights) = \
                self._request_weights("model", "eval_model_and_opponent")
            self.actor.reset_agent(self.model_id, weights)
            self.actor.reset_env(self.model_id, opponent_id, opponent_weights)


#%% core class, used as entry function


class ActorMainBase:
    def __init__(self, logger_file_dir: str = None, logger_file_level=logging.DEBUG):
        self.logger_file_dir = logger_file_dir
        self.logger_file_level = logger_file_level

    @utils.wrap_traceback
    def __call__(self, infos: Tuple[int, int, str],
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue):
        gather_id, actor_id, role = infos
        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"gather_{gather_id}_actor_{actor_id}.txt")
        utils.set_process_logger(file_path=self.logger_file_path, file_level=self.logger_file_level)
        self.main(gather_id, actor_id, role, queue_gather2actor, queue_actor2gather)

    def main(self, gather_id, actor_id, role, queue_gather2actor, queue_actor2gather):
        raise NotImplementedError


#%%


def send(queue_sender: mp.Queue, data: Tuple[str, Any],
         block: bool = True, timeout: float = None):
    queue_sender.put(data, block=block, timeout=timeout)


class GatherBase:
    """
    request latest model from model_server_conn
    request eval or sample opponent model from league_conn
    send eval or sample infos to league_conn
    send episodes to memory_server_conn
    """
    def __init__(self,
                 gather_id: int,
                 num_actors: int,
                 actor_main: ActorMainBase,
                 league_conn: connection.PickledConnection,
                 memory_server_conn: Optional[connection.PickledConnection],
                 model_server_conn: Optional[connection.PickledConnection],
                 ):

        self.gather_id = gather_id
        self.num_actors = num_actors
        self.actor_main = actor_main

        self.league_conn = league_conn
        self.memory_server_conn = memory_server_conn
        self.model_server_conn = model_server_conn

        # model_id, weights
        self.model_id = None
        self.weights = None
        self.num_sent = None
        self.max_num_sent = None
        # episodes
        self.episodes = []
        self.max_episodes_length = None
        # opponent_id, weights
        self.opponent_id = None
        self.opponent_weights = None
        self.opponent_num_sent = None
        self.max_opponent_num_sent = None  # send opponent weights only once
        # infos
        self.infos: List[Dict[str, Any]] = []
        self.max_infos_length = None
        # communicate queues
        self.queue_gather2actors = []
        self.queue_actor2gather = mp.Queue(maxsize=self.num_actors)

    def start_actors(self, role):
        # start actor process
        for i in range(self.num_actors):
            self.queue_gather2actors.append(mp.Queue(maxsize=1))
            mp.Process(target=self.actor_main, args=((self.gather_id, i, role),
                                                     self.queue_gather2actors[i],
                                                     self.queue_actor2gather),
                       name=f"gather_{self.gather_id}_actor_{i}", daemon=True).start()

    def run(self):
        raise NotImplementedError

    def _request_model_and_reset(self, cmd, data):  # model, latest
        cmd, (model_id, weights) = connection.send_recv(self.model_server_conn, (cmd, data))
        self.model_id = model_id
        self.weights = weights
        self.num_sent = 0
        logging.debug(f"send (cmd: {cmd}, data: {data}), get (cmd:{cmd}, data:{model_id}")

    def _request_opponent_and_reset(self, cmd, data):  # model, eval_opponent, sample_opponent
        cmd, (model_id, weights) = connection.send_recv(self.league_conn, (cmd, data))
        self.opponent_id = model_id
        self.opponent_weights = weights
        self.opponent_num_sent = 0
        logging.debug(f"send (cmd: {cmd}, data: {data}), get (cmd:{cmd}, data:{model_id}")

    def _request_model_and_opponent_and_reset(self, cmd, data):
        cmd, ((model_id, weights), (opponent_id, opponent_weights)) = \
            connection.send_recv(self.league_conn, (cmd, data))

        self.model_id = model_id
        self.weights = weights
        self.num_sent = 0

        self.opponent_id = opponent_id
        self.opponent_weights = opponent_weights
        self.opponent_num_sent = 0
        logging.debug(f"send (cmd: {cmd}, data: {data}), get (cmd:{cmd}, data:{model_id}, {opponent_id}")

    def _run_model(self, actor_id, cmd, data):  # model, latest, sample_opponent, eval_opponent
        if data == "latest":
            if self.num_sent >= self.max_num_sent:
                self._request_model_and_reset(cmd, data)

            send(self.queue_gather2actors[actor_id], (cmd, (self.model_id, self.weights)))
            self.num_sent += 1
            logging.debug(f"receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {self.model_id},"
                          f"num_sent: {self.num_sent}")

        elif data == "sample_opponent":
            if self.opponent_num_sent >= self.max_opponent_num_sent:
                self._request_opponent_and_reset(cmd, data)

            send(self.queue_gather2actors[actor_id], (cmd, (self.opponent_id, self.opponent_weights)))
            self.opponent_num_sent += 1
            logging.debug(f"receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {self.model_id},"
                          f"num_sent: {self.opponent_num_sent}")

        elif data == "eval_model_and_opponent":
            if self.num_sent >= self.max_num_sent:
                self._request_model_and_opponent_and_reset(cmd, data)
            send(self.queue_gather2actors[actor_id], (cmd, (
                                                            (self.model_id, self.weights),
                                                            (self.opponent_id, self.opponent_weights)
                                                            )))
            self.num_sent += 1
            self.opponent_num_sent += 1
            logging.debug(f"receive (cmd: {cmd}, data: {data}), "
                          f"send (cmd: {cmd}, data: {self.model_id}, {self.opponent_id}),"
                          f"num_sent: ({self.num_sent}, {self.opponent_num_sent})")

    def _run_infos(self, actor_id, cmd, data):  # sample_infos, eval_infos
        self.infos.append(data)
        send(self.queue_gather2actors[actor_id], (cmd, f"successfully receive {cmd}"))
        logging.debug(f"receive {cmd}, info length is {len(self.infos)}")

        if len(self.infos) >= self.max_infos_length:
            cmd, msg = connection.send_recv(self.league_conn, (cmd, self.infos))
            logging.debug(f"send cmd:{cmd}, receive msg:{msg}")
            self.infos.clear()

    def _run_episodes(self, actor_id, cmd, data):
        self.episodes.append(data)
        send(self.queue_gather2actors[actor_id], (cmd, f"successfully receive {cmd}"))
        logging.debug(f"receive {cmd}, episodes length is {len(self.episodes)}")

        if len(self.episodes) >= self.max_episodes_length:
            cmd, msg = connection.send_recv(self.memory_server_conn, (cmd, self.episodes))
            logging.debug(f"send cmd:{cmd}, response msg:{msg}")
            self.episodes.clear()


class GatherSampler(GatherBase):
    def __init__(self, gather_id: int, num_actors: int, actor_main: ActorMainBase,
                 league_conn: connection.PickledConnection,
                 memory_server_conn: connection.PickledConnection,
                 model_server_conn: connection.PickledConnection
                 ):

        super().__init__(gather_id, num_actors, actor_main,
                         league_conn, memory_server_conn, model_server_conn)

        self.max_num_sent = num_actors
        self.num_sent = self.max_num_sent

        self.max_opponent_num_sent = 1
        self.opponent_num_sent = self.max_opponent_num_sent

        self.max_episodes_length = num_actors
        self.max_infos_length = 1
        self.start_actors("sampler")

    def run(self):

        cmd2func = {
            "model": self._run_model,
            "episodes": self._run_episodes,
            "sample_infos": self._run_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)


class GatherEvaluator(GatherBase):
    def __init__(self, gather_id: int, num_actors: int, actor_main: ActorMainBase,
                 league_conn: connection.PickledConnection):

        super().__init__(gather_id, num_actors, actor_main,
                         league_conn, None, league_conn)

        self.max_num_sent = 1
        self.num_sent = self.max_num_sent

        self.max_opponent_num_sent = 1
        self.opponent_num_sent = self.max_opponent_num_sent

        self.max_infos_length = 1
        self.start_actors("evaluator")

    def run(self):
        cmd2func = {
            "model": self._run_model,
            "eval_infos": self._run_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)


#%%
AddrType = Tuple[str, int]
RoleType = Union[Literal["sampler", "evaluator"], str]


@utils.wrap_traceback
def _open_per_gather(gather_id: int,
                     role: RoleType,
                     num_actors: int,
                     actor_main: ActorMainBase,

                     league_address: AddrType,
                     memory_server_address: Optional[AddrType],
                     model_server_address: Optional[AddrType],

                     logger_file_path: str,
                     logger_file_level):
    utils.set_process_logger(file_path=logger_file_path, file_level=logger_file_level)
    # do some checks
    assert role in ["sampler", "evaluator"]
    if role == "sampler" and memory_server_address is None:
        raise ValueError("sampler actor must designate memory_server_address that used to receive episodes")
    if role == "sampler" and model_server_address is None:
        raise ValueError("sampler actor must designate model_server_address that used to request model")

    if role == "sampler":
        league_conn = connection.connect_socket_connection(*league_address)
        memory_server_conn = connection.connect_socket_connection(*memory_server_address)
        model_server_conn = connection.connect_socket_connection(*model_server_address)
        logging.info(f"successfully connected! the gather {gather_id} is starting!")
        gather = GatherSampler(gather_id=gather_id,
                               num_actors=num_actors,
                               actor_main=actor_main,

                               league_conn=league_conn,
                               memory_server_conn=memory_server_conn,
                               model_server_conn=model_server_conn,
                               )
        logging.info(f"gather {gather_id} is started!")
        gather.run()

    elif role == "evaluator":
        """
        league and model_server_conn is the same connection
        """
        league_conn = connection.connect_socket_connection(*league_address)
        gather = GatherEvaluator(gather_id=gather_id,
                                 num_actors=num_actors,
                                 actor_main=actor_main,
                                 league_conn=league_conn,
                                 )
        logging.info(f"gather {gather_id} is started!")
        gather.run()


#%%
def open_gather(num_gathers: int,
                roles: Union[RoleType, List[RoleType], Tuple[RoleType]],
                num_actors: Union[int, List[int], Tuple[int]],
                actor_main: ActorMainBase,

                league_address: Union[AddrType, List[AddrType], Tuple[AddrType]],
                memory_server_address: Union[Optional[AddrType], List[Optional[AddrType]], Tuple[Optional[AddrType]]] = None,
                model_server_address: Union[Optional[AddrType], List[Optional[AddrType]], Tuple[Optional[AddrType]]] = None,

                logger_file_dir=None,
                logger_file_level=logging.DEBUG):
    """
    :param num_gathers: 开启num_gathers个gather进程
    :param roles: 每个gather的角色， 可以传入tuple或者list为每个gather指定不同的角色
    :param num_actors:  每个进程维护num_actors个actor
    :param actor_main: 主函数，创建actor

    :param league_address: 维护eval model或者league model_weights，以及记录metrics
    :param memory_server_address: 维护memory_server地址
    :param model_server_address: 维护model_server地址，即当前learner的最新模型

    :param logger_file_dir: gather的logger_file_dir
    :param logger_file_level: whether record communication info
    """
    if isinstance(roles, str):
        roles = [roles] * num_gathers
    assert len(roles) == num_gathers

    if isinstance(num_actors, int):
        num_actors = [num_actors] * num_gathers
    assert len(num_actors) == num_gathers

    if isinstance(league_address[0], str):
        league_address = [league_address] * num_gathers
    assert len(league_address) == num_gathers

    if memory_server_address is None:
        memory_server_address = [None] * num_gathers
    if isinstance(memory_server_address[0], str):
        memory_server_address = [memory_server_address] * num_gathers
    assert len(memory_server_address) == num_gathers

    if model_server_address is None:
        model_server_address = [None] * num_gathers
    if isinstance(model_server_address[0], str):
        model_server_address = [model_server_address] * num_gathers
    assert len(model_server_address) == num_gathers

    # start gather process
    mp.set_start_method("spawn")
    processes = []

    for i in range(num_gathers):
        if logger_file_dir is not None:
            logger_file_path = os.path.join(logger_file_dir, f"gather_{i}.txt")
        else:
            logger_file_path = None
        p = mp.Process(name=f"gather_{i}",
                       target=_open_per_gather,
                       args=(i, roles[i], num_actors[i], actor_main,
                             league_address[i], memory_server_address[i], model_server_address[i],
                             logger_file_path, logger_file_level), daemon=False)
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.close()



