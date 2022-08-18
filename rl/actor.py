import bz2
import os
import pickle

import numpy as np

from rl.agent import Agent
import rl.connection as connection

from typing import Optional, Tuple, List, Dict, Union, Any, Protocol, Literal, Mapping
import logging
import multiprocessing as mp

from collections import defaultdict

from rl.utils import batchify, set_process_logger, wrap_traceback, get_element_from_batch


#%%
class Env(Protocol):
    def reset(self, *args, **kwargs) -> Any:
        ...

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict[str, Any]]:
        # discrete action space, return next_obs, reward_infos, done, truncated, info
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
                 env: Env,
                 agent: Agent,
                 # 一次采样的步数
                 num_steps: int = 0,
                 num_episodes: int = 0,
                 ):
        """
        :param env:   gym api
        :param agent: 接受batched np.ndarray 作为输入, 返回action_info, 其中必须有action字段
        :param num_steps: 采用当前模型采样num_steps步, used for sample
        :param num_episodes: 采样num_episodes个完整的episode, used for predict
        """
        self.env = env
        self.agent = agent

        self.num_steps = num_steps
        self.num_episodes = num_episodes

        self.total_episodes = 0
        self.obs = None
        self.done = None
        self.truncated = None
        self.current_episode_info = defaultdict(float)

    def reset_env(self, *args, **kwargs):
        """
        model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]
        """
        hidden = self.agent.init_hidden(batch_size=1)
        obs = self.env.reset(*args, **kwargs)
        if hidden is not None:
            self.obs = {"observation": obs, "hidden": get_element_from_batch(hidden, 0)}
        else:
            self.obs = obs
        self.done = False
        self.truncated = False
        self.current_episode_info.clear()

    def reset_agent(self, model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]):
        """ model_id = (model_name, model_index)
        assert(self.agent.model_id[0] = model_id[0])
        """
        self.agent.set_weights(model_weights, model_id[1])

    def sample(self):
        # episode generation
        if self.obs is None:
            raise RuntimeError("need reset env  in advance!")

        moment_list = []
        step = 0
        while step < self.num_steps:
            moment = dict()
            moment['observation'] = self.obs
            # sample from batch obs and get hidden and action
            action_info_batched = self.agent.sample(batchify([self.obs], unsqueeze=0))
            action_info = get_element_from_batch(action_info_batched, 0)
            moment.update(action_info)
            # step
            obs, reward_info, self.done, self.truncated, info = self.env.step(moment['action'])
            hidden = self.agent.get_hidden()
            if hidden is not None:
                self.obs = {"observation": obs, "hidden": get_element_from_batch(hidden, 0)}
            else:
                self.obs = obs
            step += 1
            moment["reward_info"] = reward_info
            moment["done"] = self.done
            moment["only_bootstrap"] = False
            moment_list.append(moment)
            # update info
            self._update_current_episode_info(reward_info)

            if self.done or self.truncated:
                if self.truncated:
                    bootstrap_moment = {
                        "observation": self.obs,
                        "reward_info": reward_info,
                        "done": True,
                        "only_bootstrap": True
                    }
                    bootstrap_moment.update(action_info)
                    moment_list.append(bootstrap_moment)
                    step += 1

                self._record_update_when_done(info)
                yield "sample_infos", self.current_episode_info  # scoring, checkpoints, model_id, opponent_id
                # need reset env outside
                if self.done or self.truncated:
                    raise RuntimeError("need call reset env outside the generator!")

        if step == self.num_steps:
            bootstrap_moment = moment_list[-1]
            bootstrap_moment["observation"] = self.obs
            bootstrap_moment["done"] = True
            bootstrap_moment["only_bootstrap"] = True
            moment_list.append(bootstrap_moment)
        yield "episodes", moment_list

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

            action_info_batched = self.agent.sample(batchify([self.obs], unsqueeze=0))
            action_info = get_element_from_batch(action_info_batched, 0)
            obs, reward_info, self.done, self.truncated, info = self.env.step(action_info['action'])
            hidden = self.agent.get_hidden()
            if hidden is not None:
                self.obs = {"observation": obs, "hidden": get_element_from_batch(hidden, 0)}
            else:
                self.obs = obs
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

    def _update_current_episode_info(self, reward_info):
        self.current_episode_info["steps"] += 1
        for key, value in reward_info.items():
            self.current_episode_info[key] += value

    def _record_update_when_done(self, env_info):
        """
        1. record according to num_episodes, current_episode_infos and ending_info
        """
        self.total_episodes += 1
        logging.info(f"num_episodes is : {self.total_episodes}, "
                     f"current episode_infos is : {self.current_episode_info}, "
                     f"ending info is {env_info}")

        meta_info = env_info
        meta_info.update(agent_id=self.agent.model_id)
        self.current_episode_info.update(meta_info=meta_info)


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


class ActorClientBase:
    def __init__(self,
                 actor_id: int,
                 actor: Actor,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 use_bz2: bool,
                 self_play: bool):

        self.actor_id = actor_id
        self.actor = actor
        self.queue_gather2actor = queue_gather2actor
        self.queue_actor2gather = queue_actor2gather
        self.use_bz2 = use_bz2
        self.self_play = self_play

    def _request_weights(self, cmd: Literal["model"], data: Literal["latest", "sample_opponent", "eval_opponent"]):
        """
        1. request weights
        """
        cmd, (model_id, weights) = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                   self.queue_gather2actor,
                                                                   self.actor_id,
                                                                   (cmd, data))
        logging.debug(f"send (cmd: {cmd}, data: {data}), receive (cmd: {cmd}, data: {model_id})")
        """
        2. decompress?
        """
        if weights is not None and self.use_bz2:
            weights = pickle.loads(bz2.decompress(weights))
        return model_id, weights

    def send_infos(self, cmd: Literal["sample_infos", "eval_infos"], data):
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   sender_id=self.actor_id,
                                                   data=(cmd, data))
        logging.debug(f"send cmd: {cmd}, receive: {msg}")

    def send_episodes(self, cmd: Literal["episodes"], data):
        if self.use_bz2:
            data = bz2.compress(pickle.dumps(data))
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   self.actor_id,
                                                   (cmd, (self.actor.agent.model_id[1], data)))
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


class ActorClientSampler(ActorClientBase):
    def _run(self):
        cmd2func = {
            "episodes": self.send_episodes_and_reset_agent,
            "sample_infos": self.send_sample_infos_and_reset_env,
        }
        model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(model_id, weights)
        self.actor.reset_env()

        sample_generator = self.actor.sample_generator()
        for cmd, data in sample_generator:
            cmd2func[cmd](cmd, data)

    def _run_self_play(self):
        cmd2func = {
            "episodes": self.send_episodes_and_reset_agent,
            "sample_infos": self.send_sample_infos_and_reset_env,
        }
        model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(model_id, weights)
        model_id, weights = self._request_weights("model", "sample_opponent")
        self.actor.reset_env(model_id, weights)

        sample_generator = self.actor.sample_generator()
        for cmd, data in sample_generator:
            cmd2func[cmd](cmd, data)

    def send_episodes_and_reset_agent(self, cmd, data):
        self.send_episodes(cmd, data)
        model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(model_id, weights)

    def send_sample_infos_and_reset_env(self, cmd, data):
        self.send_infos(cmd, data)
        if self.self_play:
            model_id, weights = self._request_weights("model", "sample_opponent")
            self.actor.reset_env(model_id, weights)
        else:
            self.actor.reset_env()


class ActorClientEvaluator(ActorClientBase):
    def _run(self):
        cmd2func = {
            "eval_infos": self.send_eval_infos_and_reset_agent_env
        }
        model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(model_id, weights)
        self.actor.reset_env()

        predict_generator = self.actor.predict_generator()
        for cmd, data in predict_generator:
            cmd2func[cmd](cmd, data)

    def _run_self_play(self):
        cmd2func = {
            "eval_infos": self.send_eval_infos_and_reset_agent_env
        }
        model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(model_id, weights)
        model_id, weights = self._request_weights("model", "eval_opponent")
        self.actor.reset_env(model_id, weights)

        predict_generator = self.actor.predict_generator()
        for cmd, data in predict_generator:
            cmd2func[cmd](cmd, data)

    def send_eval_infos_and_reset_agent_env(self, cmd, data):
        self.send_infos(cmd, data)
        model_id, weights = self._request_weights("model", "latest")
        self.actor.reset_agent(model_id, weights)
        if self.self_play:
            model_id, weights = self._request_weights("model", "eval_opponent")
            self.actor.reset_env(model_id, weights)
        else:
            self.actor.reset_env()


#%% core class, used as entry function


class ActorMainBase:
    def __init__(self, num_steps: int = 32, logger_file_dir: str = None):
        """
        the class of create actor and run sampling or predicting.
        the user should inherit this class and implement create_env_and_agent.

        :param num_steps: the sample steps of each actor, when the actor role is evaluator,
                          this parameter will be ignored
        :param logger_file_dir: the logger file directory
        """

        self.logger_file_dir = logger_file_dir
        self.logger_file_path = None
        self.num_steps = num_steps

    @wrap_traceback
    def __call__(self, infos: Tuple[int, int, str, bool, bool],
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue):
        gather_id, actor_id, actor_role, use_bz2, self_play = infos

        assert (actor_role in ["sampler", "evaluator"])

        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"gather_{gather_id}_actor_{actor_id}.txt")
        set_process_logger(file_path=self.logger_file_path)

        env, agent = self.create_env_and_agent(gather_id, actor_id)
        actor = Actor(env, agent,
                      num_steps=self.num_steps,
                      num_episodes=1,
                      )
        if actor_role == "sampler":
            actor_client = ActorClientSampler(actor_id=actor_id, actor=actor,
                                              queue_gather2actor=queue_gather2actor,
                                              queue_actor2gather=queue_actor2gather,
                                              use_bz2=use_bz2,
                                              self_play=self_play)
            actor_client.run()

        elif actor_role == "evaluator":
            actor_client = ActorClientEvaluator(actor_id=actor_id, actor=actor,
                                                queue_gather2actor=queue_gather2actor,
                                                queue_actor2gather=queue_actor2gather,
                                                use_bz2=use_bz2,
                                                self_play=self_play)
            actor_client.run()

    def create_env_and_agent(self, gather_id: int, actor_id: int):
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
                 role: Literal["sampler", "evaluator"],
                 num_actors: int,
                 league_conn: connection.PickledConnection,
                 memory_server_conn: Optional[connection.PickledConnection],
                 model_server_conn: connection.PickledConnection,
                 actor_main: ActorMainBase,
                 use_bz2,
                 self_play):
        """
        :param gather_id: the id of gather
        :param role: the role of gather.
        :param num_actors: the actors that this gather control
        :param league_conn: used to request model, send infos
        :param actor_main: func used to create actors in daemon process
        :param use_bz2: compress model and episode or not
        """
        self.gather_id = gather_id
        self.role = role
        self.num_actors = num_actors

        self.league_conn = league_conn
        self.memory_server_conn = memory_server_conn
        self.model_server_conn = model_server_conn

        self.actor_main = actor_main
        self.use_bz2 = use_bz2
        self.self_play = self_play

        # model_id, weights
        self.model_id = None
        self.weights = None
        self.num_sent = 0
        self.max_num_sent = self.num_actors

        # opponent_id, weights
        self.opponent_id = None
        self.opponent_weights = None
        self.opponent_num_sent = 0
        self.max_opponent_num_sent = 1  # send opponent weights only once

        # infos
        self.infos: List[Dict[str, Any]] = []
        self.max_infos_length = 1

        # episodes
        self.episodes = []
        self.max_episodes_length = self.num_actors

        # communicate queues
        self.queue_gather2actors = []
        self.queue_actor2gather = mp.Queue(maxsize=self.num_actors)
        # start actor process
        for i in range(self.num_actors):
            self.queue_gather2actors.append(mp.Queue(maxsize=1))
            mp.Process(target=self.actor_main, args=(
                                                     (self.gather_id, i, self.role, self.use_bz2, self.self_play),
                                                     self.queue_gather2actors[i],
                                                     self.queue_actor2gather),
                       name=f"gather_{self.gather_id}_actor_{i}", daemon=True).start()

    def run(self):
        if not self.self_play:
            self._run()
        else:
            self._run_self_play()

    def _run(self):
        raise NotImplementedError

    def _run_self_play(self):
        raise NotImplementedError

    def _request_model_and_reset(self, cmd, data):  # model, latest
        cmd, (model_id, weights) = connection.send_recv(self.model_server_conn, (cmd, data))
        self.model_id = model_id
        self.weights = weights
        self.num_sent = 0
        logging.debug(f"send (cmd: {cmd}, data: {data}), get (cmd:{cmd}, data:{model_id}")

    def _request_opponent_model_and_reset(self, cmd, data):  # model, eval_opponent, sample_opponent
        cmd, (model_id, weights) = connection.send_recv(self.league_conn, (cmd, data))
        self.opponent_id = model_id
        self.opponent_weights = weights
        self.opponent_num_sent = 0
        logging.debug(f"send (cmd: {cmd}, data: {data}), get (cmd:{cmd}, data:{model_id}")

    def _run_model(self, actor_id, cmd, data):  # model, latest, sample_opponent, eval_opponent
        if data == "latest":
            if self.num_sent >= self.max_num_sent:
                self._request_model_and_reset(cmd, data)

            send(self.queue_gather2actors[actor_id], (cmd, (self.model_id, self.weights)))
            self.num_sent += 1
            logging.debug(f" receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {self.model_id},"
                          f"num_sent: {self.num_sent}")

        elif data in ["sample_opponent", "eval_opponent"]:
            if self.opponent_num_sent >= self.max_opponent_num_sent:
                self._request_opponent_model_and_reset(cmd, data)

            send(self.queue_gather2actors[actor_id], (cmd, (self.opponent_id, self.opponent_weights)))
            self.opponent_num_sent += 1
            logging.debug(f" receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {self.model_id},"
                          f"num_sent: {self.opponent_num_sent}")

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
    def _run(self):
        self._request_model_and_reset("model", "latest")

        cmd2func = {
            "model": self._run_model,
            "episodes": self._run_episodes,
            "sample_infos": self._run_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)

    def _run_self_play(self):
        self._request_model_and_reset("model", "latest")
        self._request_opponent_model_and_reset("model", "sample_opponent")

        cmd2func = {
            "model": self._run_model,
            "episodes": self._run_episodes,
            "sample_infos": self._run_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)


class GatherEvaluator(GatherBase):
    def _run(self):
        self._request_model_and_reset("model", "latest")

        cmd2func = {
            "model": self._run_model,
            "eval_infos": self._run_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)

    def _run_self_play(self):
        self._request_model_and_reset("model", "latest")
        self._request_opponent_model_and_reset("model", "eval_opponent")

        cmd2func = {
            "model": self._run_model,
            "eval_infos": self._run_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)


#%%
AddrType = Tuple[str, int]
RoleType = Literal["sampler", "evaluator"]


@wrap_traceback
def _open_per_gather(gather_id: int,
                     role: RoleType,
                     num_actors: int,
                     league_address: AddrType,
                     memory_server_address: Optional[AddrType],
                     model_server_address: Optional[AddrType],
                     actor_main: ActorMainBase,
                     use_bz2: bool,
                     self_play: bool,
                     logger_file_path: str):
    set_process_logger(file_path=logger_file_path)
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
                               role = role,
                               num_actors=num_actors,
                               league_conn=league_conn,
                               memory_server_conn=memory_server_conn,
                               model_server_conn=model_server_conn,
                               actor_main=actor_main,
                               use_bz2=use_bz2,
                               self_play=self_play)
        logging.info(f"gather {gather_id} is started!")
        gather.run()

    elif role == "evaluator":
        """
        league and model_server_conn is the same connection
        """
        league_conn = connection.connect_socket_connection(*league_address)
        gather = GatherEvaluator(gather_id=gather_id,
                                 role="evaluator",
                                 num_actors=num_actors,
                                 league_conn=league_conn,
                                 memory_server_conn=None,
                                 model_server_conn=league_conn,
                                 actor_main=actor_main,
                                 use_bz2=use_bz2,
                                 self_play=self_play)
        logging.info(f"gather {gather_id} is started!")
        gather.run()


#%%
def open_gather(num_gathers: int,
                roles: Union[RoleType, List[RoleType], Tuple[RoleType]],
                num_actors: Union[int, List[int], Tuple[int]],
                league_address: Union[AddrType, List[AddrType], Tuple[AddrType]],
                memory_server_address: Union[Optional[AddrType], List[Optional[AddrType]], Tuple[Optional[AddrType]]],
                model_server_address: Union[Optional[AddrType], List[Optional[AddrType]], Tuple[Optional[AddrType]]],
                actor_main: ActorMainBase,
                use_bz2: bool = True,
                self_play: bool = False,
                logger_file_dir=None):
    """
    :param num_gathers: 开启num_gathers个gather进程
    :param roles: 每个gather的角色， 可以传入tuple或者list为每个gather指定不同的角色
    :param num_actors:  每个进程维护num_actors个actor
    :param league_address: 维护eval model或者league models，以及记录metrics
    :param memory_server_address: 维护memory_server地址
    :param model_server_address: 维护model_server地址，即当前learner的最新模型
    :param actor_main: 主函数，创建actor
    :param use_bz2: 是否压缩episodes和model weights
    :param self_play: 是否self-play, 即request opponent
    :param logger_file_dir: gather的logger_file_dir
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
                       args=(i, roles[i], num_actors[i],
                             league_address[i], memory_server_address[i], model_server_address[i],
                             actor_main, use_bz2, self_play, logger_file_path), daemon=False)
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.close()



