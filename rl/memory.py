import logging
import torch
import queue
import multiprocessing as mp
import threading

from typing import Union, List, Dict, Tuple
from tensorboardX import SummaryWriter

import rl.connection as connection
import rl.utils as utils

import pickle
import bz2
import numpy as np
from collections import defaultdict

EpisodeType = Tuple[int, List[Dict]]  # int 表示采用第几次的模型采样的数据, List[Dict] 表示采样的episode

CompressedEpisodeType = Tuple[int, bytes]  # int含义同上, bytes表示用bz2压缩后的采样的episode


class BatchMaker:
    def __init__(self, decompress=False, to_tensor=False, device=torch.device("cpu")):
        self.decompress = decompress
        self.to_tensor = to_tensor
        self.device = device

    def __call__(self, episodes: List[Union[EpisodeType, CompressedEpisodeType]]) -> Tuple[int, Dict]:
        model_indexes = [index for index, episode in episodes]
        episodes = [episode for index, episode in episodes]
        if self.decompress:
            episodes = [pickle.loads(bz2.decompress(episode)) for episode in episodes]

        episodes = [utils.batchify(episode, unsqueeze=0) for episode in episodes]
        episodes = utils.batchify(episodes, unsqueeze=0)

        if self.to_tensor:
            episodes = utils.to_tensor(episodes, unsqueeze=None, device=self.device)

        return int(np.mean(model_indexes)), episodes

"""
def make_batch(episodes: List[EpisodeType], to_tensor=True) -> Tuple[int, Dict]:  # batch_size*time_step
    model_indexes = [index for index, episode in episodes]
    episodes = [episode for index, episode in episodes]

    episodes = [utils.batchify(episode, unsqueeze=0) for episode in episodes]
    episodes = utils.batchify(episodes, unsqueeze=0)

    if to_tensor:
        episodes = utils.to_tensor(episodes, unsqueeze=None, device=torch.device("cpu"))
    return int(np.mean(model_indexes)), episodes
    
    

def decompress_and_make_batch(episodes: List[CompressedEpisodeType], to_tensor=True) -> Tuple[int, Dict]:
    model_indexes = [index for index, episode in episodes]
    episodes = [episode for index, episode in episodes]

    episodes = [utils.batchify(pickle.loads(bz2.decompress(episode)), unsqueeze=0) for episode in episodes]
    episodes = utils.batchify(episodes, unsqueeze=0)

    if to_tensor:
        episodes = utils.to_tensor(episodes, unsqueeze=None, device=torch.device("cpu"))
    return int(np.mean(model_indexes)), episodes
"""


class MemoryReplayBase:
    """
       MemoryServer is the data structure that caches the episode sent from actors
    """
    def __init__(self, queue_sender: mp.Queue, use_bz2, to_tensor, device):
        """
        at least have queue_receiver
        :param queue_sender: the queue that caches batched np.ndarray
        :param use_bz2: whether each episode is compressed by bz2
        """
        self.queue_sender = queue_sender
        self.use_bz2 = use_bz2
        self.to_tensor = to_tensor
        self.device = device

    def cache(self, episodes: Union[List[EpisodeType], List[CompressedEpisodeType]]) -> None:
        """
        the api that caches the episode
        :param episodes: list of episode,
               episode itself is a list of dict,
               dict contains obs, action, reward, done
        :return: None
        """
        raise NotImplementedError

    def start(self):
        """
        the logic that batchifies the cached raw episodes
        when asynchronous, the memory_server will call this function only once
        when synchronous,  the memory_server will call this function many times.
        :return: None
        """
        raise NotImplementedError

    def stop(self):
        pass


class TrajList(MemoryReplayBase):
    """
    used for a2c algorithm
    """
    def __init__(self, queue_sender, use_bz2=False, to_tensor=True, device=torch.device("cpu")):
        super().__init__(queue_sender, use_bz2, to_tensor, device)
        self.episode_list = []
        self.num_cashed = 0
        self.num_sent = 0

        self.batch_maker = BatchMaker(decompress=use_bz2, to_tensor=to_tensor, device=device)

    def cache(self, episodes: Union[List[EpisodeType], List[CompressedEpisodeType]]) -> None:
        self.episode_list.extend(episodes)
        self.num_cashed += len(episodes)
        logging.debug(f"total cached data num is {self.num_cashed}")

    def start(self):
        batch = self.batch_maker(self.episode_list)
        connection.send_with_stop_flag(self.queue_sender, is_stop=False, data=batch)  # False 表示发送端没有停止发送
        self.num_sent += 1
        logging.debug(f"total sent data num is {self.num_sent}")
        self.episode_list.clear()

    def stop(self):
        connection.send_with_stop_flag(self.queue_sender, is_stop=True, data=None)  # False 表示发送端没有停止发送
        logging.debug(f'successfully stop sending data')


class TrajQueue(MemoryReplayBase):
    def __init__(self,
                 maxlen: int,
                 queue_sender: mp.Queue,
                 batch_size=64,
                 use_bz2: bool = True,
                 to_tensor: bool = True,
                 device=torch.device("cpu")
                 ):
        super().__init__(queue_sender, use_bz2, to_tensor, device)
        self.episode_queue = queue.Queue(maxsize=maxlen)
        self.batch_size = batch_size

        self.batch_maker = BatchMaker(decompress=use_bz2, to_tensor=to_tensor, device=device)

        self.num_cached = 0
        self.is_stop = False

    def cache(self, episodes: Union[List[EpisodeType], List[CompressedEpisodeType]]) -> None:
        for episode in episodes:
            while True:
                try:
                    self.episode_queue.put(episode, timeout=0.1)
                    break
                except queue.Full:
                    logging.debug("the queue is full")
            self.num_cached += 1
        logging.debug(f"total cashed data num is {self.num_cached}")

    def send_raw_batch(self):
        while True:
            raw_batch = []
            for _ in range(self.batch_size):
                while True:
                    try:
                        raw_batch.append(self.episode_queue.get(timeout=0.1))
                        break
                    except queue.Empty:
                        if self.is_stop:
                            return
            yield raw_batch

    def start(self):
        threading.Thread(target=self._make_batch, args=(), name="batch_maker", daemon=True).start()

    @utils.wrap_traceback
    def _make_batch(self):
        num = 0
        send_generator = self.send_raw_batch()
        try:
            while True:
                batched = self.batch_maker(next(send_generator))
                connection.send_with_stop_flag(self.queue_sender, False, batched)
                num += 1
                logging.debug(f"successfully make and send batch num: {num}")
        except StopIteration:
            connection.send_with_stop_flag(self.queue_sender, True, (-1, None))
            logging.info(f"successfully stop send!")

    def stop(self):
        self.is_stop = True


class TrajQueueMP(TrajQueue):
    def __init__(self,
                 maxlen: int,
                 queue_sender: mp.Queue,
                 batch_size: int = 64,
                 use_bz2: bool = True,
                 to_tensor: bool = True,
                 device=torch.device("cpu"),
                 num_batch_maker: int = 2,
                 logger_file_dir: str = None,
                 ):
        super().__init__(maxlen, queue_sender, batch_size, use_bz2, to_tensor, device)

        self.batch_maker = connection.MultiProcessJobExecutors(func=self.batch_maker,
                                                               send_generator=self.send_raw_batch(),
                                                               num=num_batch_maker,
                                                               queue_receiver=self.queue_sender,
                                                               name_prefix="batch_maker",
                                                               logger_file_dir=logger_file_dir)

    def start(self):
        self.batch_maker.start()


class MemoryServer:
    def __init__(self,
                 memory_replay: MemoryReplayBase,
                 port: int,
                 actor_num=None,
                 tensorboard_dir=None
                 ):

        self.actor_communicator = connection.QueueCommunicator(port, actor_num)
        self.memory_replay = memory_replay

        if tensorboard_dir is not None:
            self.sw = SummaryWriter(logdir=tensorboard_dir)

        self.num_sample_info_received = defaultdict(lambda: 0)
        self.actor_num = actor_num

    def run(self):
        logging.info("start server to receive episodes that generated by actor")
        self.actor_communicator.run()
        self.memory_replay.start()
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            """
            cmd: Literal["episode", "sample_infos"]
            """
            logging.debug(cmd)
            if cmd == "episodes":
                self.memory_replay.cache(data)
                self.actor_communicator.send(conn, (cmd, "successfully receive episodes"))

            elif cmd == "sample_infos":  # deprecated!
                self._record_sample_info(data)
                self.actor_communicator.send(conn, (cmd, "successfully record sample info"))

    def run_sync(self):
        self.actor_communicator.run_sync()

        conns = []
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            """
            cmd: Literal["episode", "sample_info"]
            """
            logging.debug(cmd)
            if cmd == "episodes":
                self.memory_replay.cache(data)
                conns.append(conn)

                if len(conns) == self.actor_num:
                    self.memory_replay.start()
                    for conn in conns:
                        self.actor_communicator.send(conn, (cmd, "successfully receive episodes"))
                    conns = []

            elif cmd == "sample_infos":
                self._record_sample_info(data)
                self.actor_communicator.send(conn, (cmd, "successfully record sample infos"))

    def _record_sample_info(self, data: Dict[str, list]):
        for key, values in data.items():
            for value in values:
                self.num_sample_info_received[key] += 1
                self.sw.add_scalar(tag=key, scalar_value=value, global_step=self.num_sample_info_received[key])






