import bz2
import os
import pickle

import numpy as np

from rl.agent import Agent
import rl.connection as connection

from tensorboardX import SummaryWriter
from typing import Optional, Tuple, List, Dict, Union, Any, Protocol, Literal, Mapping
import logging
import multiprocessing as mp

from collections import defaultdict

from rl.utils import batchify, set_process_logger, wrap_traceback

__all__ = ["Actor", "ActorCreateBase", "open_gather"]


class Env(Protocol):
    def reset(self, model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]) -> Any:  # return obs
        ...

    def step(self, action: int) -> Tuple[Any, Dict[str, float], bool, Dict[str, Any]]:
        # discrete action space, return next_obs, reward_infos, done, info
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
    def __init__(self, env: Env, agent: Agent,
                 num_steps: int = 0,
                 num_episodes: int = 0,
                 get_full_episode: bool = False,
                 tensorboard_dir: str = None
                 ):
        """
        一次采样至少num_steps步，
        至少采样num_episodes个完整的episode,在steps步之后，
        若get_full_episode等于True,继续采样一个完整的episode

        :param env:   gym api
        :param agent: 接受batched np.array 作为输入, 返回action_info, 其中必须有action字段
        :param num_steps: 采用当前模型采样num_steps步
        :param num_episodes: 采样num_episodes个完整的episode
        :param get_full_episode: 若采样steps步之后，继续采样一定步数直到当前episode结束
        """
        self.env = env
        self.agent = agent

        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.get_full_episode = get_full_episode

        self.total_episodes = 0
        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)

        self.obs = None
        self.done = None
        self.current_episode_infos = None

    def reset_env(self, model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]):
        self.obs = self.env.reset(model_id, model_weights)
        self.done = False
        self.current_episode_infos = defaultdict(int)

    def reset_agent(self, model_id: Tuple[str, int], model_weights: Mapping[str, np.ndarray]):
        """ model_id = (model_name, model_index)
        assert(self.agent.model_id[0] = model_id[0])
        """
        self.agent.set_weights(model_weights, model_id[1])

    def sample(self):
        # episode generation
        if self.obs is None:
            raise RuntimeError("need call reset env advance!")

        num_episodes = 0
        episode = []
        step = 0

        while step < self.num_steps or num_episodes < self.num_episodes or (not self.done and self.get_full_episode):

            moment = dict()
            moment['observation'] = self.obs

            # batch obs
            action_info = self.agent.sample(batchify([self.obs], unsqueeze=0))
            for key, value in action_info.items():
                moment[key] = value[0]

            self.obs, reward_infos, self.done, info = self.env.step(moment['action'])
            step += 1

            moment["reward_infos"] = reward_infos
            moment["done"] = self.done
            episode.append(moment)

            self._update_current_episode_infos(reward_infos)

            if self.done:
                self._record_update_when_done(info)
                yield "sample_infos", self.current_episode_infos  # scoring, checkpoints, model_id, opponent_id
                # need reset env outside
                if self.done:
                    raise RuntimeError("need call reset env outside the generator!")

                num_episodes += 1
                if step >= self.num_steps and num_episodes >= self.num_episodes:
                    break

        yield "episodes", episode

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
        step = 0

        while step < self.num_steps or num_episodes < self.num_episodes or (not self.done and self.get_full_episode):

            action_info = self.agent.predict(batchify([self.obs], unsqueeze=0))
            self.obs, reward_infos, self.done, info = self.env.step(action_info['action'][0])
            step += 1

            self._update_current_episode_infos(reward_infos)

            if self.done:
                self._record_update_when_done(info)
                yield "eval_infos", self.current_episode_infos  # scoring, checkpoints, model_id, opponent_id
                if self.done:
                    raise RuntimeError("need call reset env outside the generator!")

                num_episodes += 1
                if step >= self.num_steps and num_episodes >= self.num_episodes:
                    break

    def predict_generator(self):
        """
        infinitely predict
        """
        while True:
            predict = self.predict()
            for cmd, data in predict:
                yield cmd, data

    def _update_current_episode_infos(self, reward_infos):
        self.current_episode_infos["steps"] += 1
        for key, value in reward_infos.items():
            self.current_episode_infos[key] += value

    def _record_update_when_done(self, ending_info):
        """
        1. record according to num_episodes, current_episode_infos and ending_info
        """
        self.total_episodes += 1
        logging.info(f"num_episodes is : {self.total_episodes}, "
                     f"current episode_infos is : {self.current_episode_infos}, "
                     f"ending info is {ending_info}")

        if hasattr(self, "sw"):
            for key, value in self.current_episode_infos.items():
                self.sw.add_scalar(key, value, self.total_episodes)

        if hasattr(self.agent, "model_id"):
            self.current_episode_infos["model_id"] = self.agent.model_id

        if "win" in ending_info.keys():
            self.current_episode_infos["win"] = ending_info["win"]

        if "opponent_id" in ending_info.keys():
            self.current_episode_infos["opponent_id"] = ending_info["opponent_id"]


#%%

def send_with_sender_id_and_receive(queue_sender: mp.Queue, queue_receiver: mp.Queue,
                                    sender_id: int, data: Tuple[str, Any],
                                    block: bool = True, timeout: float = None) -> Tuple[str, Any]:
    queue_sender.put((sender_id, data), block=block, timeout=timeout)
    cmd, data = queue_receiver.get()
    return cmd, data


class ActorClientBase:
    def __init__(self, actor_id: int, actor: Actor,
                 queue_gather2actor: mp.Queue,
                 queue_actor2gather: mp.Queue,
                 use_bz2: bool):

        self.actor_id = actor_id
        self.actor = actor
        self.queue_gather2actor = queue_gather2actor
        self.queue_actor2gather = queue_actor2gather
        self.use_bz2 = use_bz2

    def _request_weights(self, data: Literal["latest", -1, "sample_opponent", "eval_opponent"]) -> \
            Tuple[Tuple[str, int], Mapping[str, np.ndarray]]:
        """
        1. request weights
        """
        cmd, (model_id, weights) = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                                   self.queue_gather2actor,
                                                                   self.actor_id, ("model", data))
        logging.debug(f"successfully send cmd: model, data: {data} and receive model id: {model_id}")
        """
        2. decompress? and set weights
        """
        if weights is not None and self.use_bz2:
            weights = pickle.loads(bz2.decompress(weights))

        return model_id, weights

    def run(self):
        raise NotImplementedError


class ActorClientSampler(ActorClientBase):
    def run(self):
        cmd2func = {
            "sample_infos": self.send_sample_infos_and_request_opponent,
            "episodes": self.send_episodes_and_request_agent
        }

        model_id, weights = self._request_weights("sample_opponent")
        self.actor.reset_env(model_id, weights)

        model_id, weights = self._request_weights("latest")
        self.actor.reset_agent(model_id, weights)

        sample_generator = self.actor.sample_generator()

        for cmd, data in sample_generator:
            cmd2func[cmd](cmd, data)

    def send_sample_infos_and_request_opponent(self, cmd, data):
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   sender_id=self.actor_id,
                                                   data=(cmd, data))
        logging.debug(f"send cmd: {cmd}, response: {msg}")

        model_id, weights = self._request_weights("sample_opponent")
        self.actor.reset_env(model_id, weights)

    def send_episodes_and_request_agent(self, cmd, data):
        if self.use_bz2:
            data = bz2.compress(pickle.dumps(data))
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   self.actor_id,
                                                   (cmd, (self.actor.agent.model_id[1], data)))
        logging.debug(f"send cmd: {cmd}, response: {msg}")

        model_id, weights = self._request_weights("latest")
        self.actor.reset_agent(model_id, weights)


class ActorClientEvaluator(ActorClientBase):
    def run(self):
        cmd2func = {
            "eval_infos": self.send_eval_infos_and_request_agent_and_opponent
        }

        model_id, weights = self._request_weights("eval_opponent")
        self.actor.reset_env(model_id, weights)

        model_id, weights = self._request_weights(-1)
        self.actor.reset_agent(model_id, weights)

        predict_generator = self.actor.predict_generator()

        for cmd, data in predict_generator:
            cmd2func[cmd](cmd, data)

    def send_eval_infos_and_request_agent_and_opponent(self, cmd, data):
        cmd, msg = send_with_sender_id_and_receive(self.queue_actor2gather,
                                                   self.queue_gather2actor,
                                                   sender_id=self.actor_id,
                                                   data=(cmd, data))
        logging.debug(f"send cmd: {cmd}, response: {msg}")

        model_id, weights = self._request_weights(-1)
        self.actor.reset_agent(model_id, weights)

        model_id, weights = self._request_weights("eval_opponent")
        self.actor.reset_env(model_id, weights)

#%% core class, used as entry function


class ActorCreateBase:
    def __init__(self, num_steps: int = 50, num_episodes: int = 0, get_full_episode: bool = False,
                 logger_file_dir: str = None,):
        """
        the class of create actor and run sampling or predicting.
        the user should inherit this class and implement create_env_and_agent.
        :param logger_file_dir: the logger file directory
        :param num_steps: the sample steps of each actor, when the actor role is evaluator,
                          this parameter will be ignored
        :param num_episodes: the episodes num of each actor, when the actor role is evaluator,
                             this parameter will be ignored
        :param get_full_episode: the flag of whether to get full episode of each actor,
                                 when the actor role is evaluator,
                                 this parameter will be ignored
        """

        self.logger_file_dir = logger_file_dir
        self.logger_file_path = None

        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.get_full_episode = get_full_episode

    def __call__(self, infos: Tuple[int, int, str, bool], queue_gather2actor: mp.Queue, queue_actor2gather: mp.Queue):
        gather_id, actor_id, actor_role, use_bz2 = infos

        assert (actor_role in ["sampler", "evaluator"])

        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"gather_{gather_id}_actor_{actor_id}.txt")
        set_process_logger(file_path=self.logger_file_path)

        env, agent = self.create_env_and_agent(gather_id, actor_id)

        if actor_role == "sampler":
            actor = Actor(env, agent,
                          num_steps=self.num_steps,
                          num_episodes=self.num_episodes,
                          get_full_episode=self.get_full_episode)

            actor_client = ActorClientSampler(actor_id=actor_id, actor=actor,
                                              queue_gather2actor=queue_gather2actor,
                                              queue_actor2gather=queue_actor2gather,
                                              use_bz2=use_bz2)
            actor_client.run()

        elif actor_role == "evaluator":
            actor = Actor(env, agent,
                          num_steps=0,
                          num_episodes=1,
                          get_full_episode=True)  # 每次采样1个episode

            actor_client = ActorClientEvaluator(actor_id=actor_id, actor=actor,
                                                queue_gather2actor=queue_gather2actor,
                                                queue_actor2gather=queue_actor2gather,
                                                use_bz2=use_bz2)
            actor_client.run()

    def create_env_and_agent(self, gather_id: int, actor_id: int):
        raise NotImplementedError


#%%

def send(queue_sender: mp.Queue, data: Tuple[str, Any],
         block: bool = True, timeout: float = None):
    queue_sender.put(data, block=block, timeout=timeout)


class GatherBase:
    def __init__(self, gather_id: int, num_actors: int,
                 league_conn: connection.PickledConnection,
                 func: ActorCreateBase, use_bz2=True):
        """
        :param gather_id: the id of gather
        :param num_actors: the actors that this gather control
        :param league_conn: used to request model and send infos
        :param func: func used to create actors in daemon process
        :param use_bz2: compress model and episode or not
        """
        self.gather_id = gather_id
        self.num_actors = num_actors
        self.league_conn = league_conn
        self.func = func
        self.use_bz2 = use_bz2

        # model_id, weights
        self.model_id = None
        self.weights = None
        self.num_sent = 0
        self.max_num_sent = self.num_actors

        # opponent_id, weights
        self.opponent_id = None
        self.opponent_weights = None
        self.opponent_num_sent = 0

        # infos
        self.infos: List[Dict[str, Any]] = []
        self.max_infos_length = self.num_actors

        # communicate queues
        self.queue_gather2actors = []
        self.queue_actor2gather = mp.Queue(maxsize=self.num_actors)

    def _create_actors(self, actor_role):
        for i in range(self.num_actors):
            self.queue_gather2actors.append(mp.Queue(maxsize=1))
            mp.Process(target=self.func, args=((self.gather_id, i, actor_role, self.use_bz2),
                                               self.queue_gather2actors[i],
                                               self.queue_actor2gather),
                       name=f"gather_{self.gather_id}_actor_{i}", daemon=True).start()

    def run(self):
        raise NotImplementedError

    def _request_model_and_set(self, cmd, data):
        _, (model_id, weights) = connection.send_recv(self.league_conn, (cmd, data))
        self.model_id = model_id
        self.weights = weights
        self.num_sent = 0
        logging.debug(f"send cmd {cmd}, data {data}, get model_id {model_id}")

    def _request_opponent_model_and_set(self, cmd, data):
        _, (model_id, weights) = connection.send_recv(self.league_conn, (cmd, data))
        self.opponent_id = model_id
        self.opponent_weights = weights
        self.opponent_num_sent = 0
        logging.debug(f"send cmd {cmd}, data {data}, get model_id {model_id}")


class GatherSampler(GatherBase):
    def __init__(self, gather_id: int, num_actors: int,
                 league_conn: connection.PickledConnection,
                 memory_server_conn: connection.PickledConnection,
                 func: ActorCreateBase,
                 use_bz2=True
                 ):
        super().__init__(gather_id, num_actors, league_conn, func, use_bz2)
        self.actor_role = "sampler"
        self.memory_server_conn = memory_server_conn  # send episodes and sample_infos
        # episodes
        self.episodes = []
        self.max_episodes_length = self.num_actors

        self._create_actors(self.actor_role)

    def run(self):
        self._request_model_and_set("model", "latest")
        self._request_opponent_model_and_set("model", "sample_opponent")

        cmd2func = {
            "model": self._run_model,
            "episodes": self._run_episodes,
            "sample_infos": self._run_sample_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)

    def _run_model(self, actor_id, cmd, data):
        logging.debug(f"cmd is {cmd}, data is {data}")

        if data == "latest":
            if self.num_sent >= self.max_num_sent:
                self._request_model_and_set(cmd, data)

            send(self.queue_gather2actors[actor_id], (cmd, (self.model_id, self.weights)))
            self.num_sent += 1

        elif data == "sample_opponent":
            if self.opponent_num_sent >= self.max_num_sent:
                self._request_opponent_model_and_set(cmd, data)

            send(self.queue_gather2actors[actor_id], (cmd, (self.opponent_id, self.opponent_weights)))
            self.opponent_num_sent += 1

    def _run_episodes(self, actor_id, cmd, data):
        self.episodes.append(data)

        if len(self.episodes) >= self.max_episodes_length:
            cmd, msg = connection.send_recv(self.memory_server_conn, (cmd, self.episodes))
            logging.debug(f"send cmd:{cmd}, response msg:{msg}")
            self.episodes.clear()

        send(self.queue_gather2actors[actor_id], (cmd, "successfully receive episodes"))

    def _run_sample_infos(self, actor_id, cmd, data):
        self.infos.append(data)

        if len(self.infos) >= self.max_infos_length:
            cmd, msg = connection.send_recv(self.league_conn, (cmd, self.infos))
            logging.debug(f"cmd:{cmd}, msg:{msg}")
            self.infos.clear()

        send(self.queue_gather2actors[actor_id], (cmd, "successfully receive sample_infos"))


class GatherEvaluator(GatherBase):
    def __init__(self, gather_id: int, num_actors: int,
                 league_conn: connection.PickledConnection,
                 func: ActorCreateBase,
                 use_bz2=True
                 ):
        super().__init__(gather_id, num_actors, league_conn, func, use_bz2)
        self.actor_role = "evaluator"
        self._create_actors(self.actor_role)

    def run(self):
        self._request_model_and_set("model", -1)
        self._request_opponent_model_and_set("model", "eval_opponent")

        cmd2func = {
            "model": self._run_model,
            "eval_infos": self._run_eval_infos
        }
        while True:
            actor_id, (cmd, data) = self.queue_actor2gather.get()
            cmd2func[cmd](actor_id, cmd, data)

    def _run_model(self, actor_id: int, cmd: Literal["model"], data: Literal[-1, "eval_opponent"]):
        logging.debug(f"cmd is {cmd}, data is {data}")
        if data == -1:
            if self.num_sent >= self.max_num_sent:
                self._request_model_and_set(cmd, data)
            send(self.queue_gather2actors[actor_id], (cmd, (self.model_id, self.weights)))
            self.num_sent += 1

        elif data == "eval_opponent":
            if self.opponent_num_sent >= self.max_num_sent:
                self._request_opponent_model_and_set(cmd, data)
            send(self.queue_gather2actors[actor_id], (cmd, (self.opponent_id, self.opponent_weights)))
            self.opponent_num_sent += 1

    def _run_eval_infos(self, actor_id: int, cmd: Literal["eval_infos"], data: Dict[str, Any]):
        self.infos.append(data)

        if len(self.infos) >= self.max_infos_length:
            cmd, msg = connection.send_recv(self.league_conn, (cmd, self.infos))
            logging.debug(f"cmd:{cmd}, msg:{msg}")
            self.infos.clear()

        send(self.queue_gather2actors[actor_id], (cmd, "successfully receive eval_infos"))


def _open_per_gather(gather_id: int,
                     memory_server_address: Optional[Tuple[str, int]],
                     league_address: Tuple[str, int],
                     num_actors: int, actor_role: str,
                     func: ActorCreateBase,
                     use_bz2: bool,
                     logger_file_path: str):

    assert (actor_role in ["sampler", "evaluator"])
    if actor_role == "sampler" and memory_server_address is None:
        raise ValueError("sampler actor must designate memory_server_address that used to receive episodes")

    set_process_logger(file_path=logger_file_path)

    if memory_server_address is not None:
        memory_server_conn = connection.connect_socket_connection(*memory_server_address)
    else:
        memory_server_conn = None

    league_conn = connection.connect_socket_connection(*league_address)

    logging.info(f"successfully connected! the gather {gather_id} is starting!")

    if actor_role == "sampler":
        gather = GatherSampler(gather_id=gather_id, num_actors=num_actors, league_conn=league_conn,
                               memory_server_conn=memory_server_conn, func=func, use_bz2=use_bz2)
        logging.info(f"gather {gather_id} is started!")
        gather.run()

    elif actor_role == "evaluator":
        gather = GatherEvaluator(gather_id=gather_id, num_actors=num_actors, league_conn=league_conn,
                                 func=func, use_bz2=use_bz2)
        logging.info(f"gather {gather_id} is started!")
        gather.run()


AddrType = Tuple[str, int]


def open_gather(num_gathers: int,
                memory_server_address: Union[Optional[AddrType], List[Optional[AddrType]], Tuple[Optional[AddrType]]],
                league_address: Union[AddrType, List[AddrType], Tuple[AddrType]],
                num_actors: Union[int, List[int], Tuple[int]],
                actor_roles: Union[str, List[str], Tuple[str]],
                func: ActorCreateBase,
                use_bz2: bool = True,
                logger_file_dir=None):

    if memory_server_address is None:
        memory_server_address = [None] * num_gathers

    if isinstance(memory_server_address[0], str):
        memory_server_address = [memory_server_address] * num_gathers

    if isinstance(league_address[0], str):
        league_address = [league_address] * num_gathers

    if isinstance(num_actors, int):
        num_actors = [num_actors] * num_gathers

    if isinstance(actor_roles, str):
        actor_roles = [actor_roles] * num_gathers

    mp.set_start_method("spawn")
    processes = []

    for i in range(num_gathers):
        if logger_file_dir is not None:
            logger_file_path = os.path.join(logger_file_dir, f"gather_{i}.txt")
        else:
            logger_file_path = None
        p = mp.Process(name=f"gather_{i}",
                       target=_open_per_gather,
                       args=(i, memory_server_address[i], league_address[i],
                             num_actors[i], actor_roles[i],
                             func, use_bz2, logger_file_path), daemon=False)
        processes.append(p)

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.close()



