import logging
import random

import torch
import queue
import multiprocessing as mp
import threading

from typing import Union, List, Dict, Tuple


import rl.connection as connection
import rl.utils as utils

import pickle
import bz2
import numpy as np
from collections import deque


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
                 # batch_maker args
                 num_batch_maker: int = 2,
                 logger_file_dir: str = None,
                 logger_file_level=logging.DEBUG
                 ):
        super().__init__(maxlen, queue_sender, batch_size, use_bz2, to_tensor, device)

        self.batch_maker = connection.MultiProcessJobExecutors(func=self.batch_maker,
                                                               send_generator=self.send_raw_batch(),
                                                               num=num_batch_maker,
                                                               queue_receiver=self.queue_sender,
                                                               name_prefix="batch_maker",
                                                               logger_file_dir=logger_file_dir,
                                                               file_level=logger_file_level)

    def start(self):
        self.batch_maker.start()


#%%
ChunkType = Tuple[int, bytes, bytes]


class SelectBatcher:
    def __init__(self,  to_tensor=False, device=torch.device("cpu")):
        self.to_tensor = to_tensor
        self.device = device

    def __call__(self, compressed_chunks: List[ChunkType]) -> Tuple[int, Dict]:
        indexes = []
        chunks = []
        for compressed_chunk in compressed_chunks:
            index, chunk = self._make(compressed_chunk)
            indexes.append(index)
            chunks.append(chunk)

        chunks = utils.batchify(chunks, unsqueeze=0)

        if self.to_tensor:
            chunks = utils.to_tensor(chunks, unsqueeze=None, device=self.device)

        return int(np.mean(indexes)), chunks

    @staticmethod
    def _make(compressed_chunk: ChunkType) -> Tuple[int, Dict]:
        index, chunk1, chunk2 = compressed_chunk
        chunk1 = pickle.loads(bz2.decompress(chunk1))
        chunk_length = len(chunk1)

        chunk = []
        chunk.extend(chunk1)
        chunk.extend(pickle.loads(bz2.decompress(chunk2)))

        start = random.randrange(0, len(chunk) - chunk_length)
        chunk = chunk[start: start+chunk_length+1]
        chunk[-1]["only_bootstrap"] = True

        chunk = utils.batchify(chunk, unsqueeze=0)

        return index, chunk


class TrajListMP(MemoryReplayBase):
    def __init__(self,
                 maxlen: int,
                 queue_sender: mp.Queue,
                 batch_size: int = 64,
                 priority_replay: bool = False,
                 to_tensor: bool = True,
                 device=torch.device("cpu"),
                 # batch_maker args
                 num_batch_maker: int = 2,
                 logger_file_dir: str = None,
                 logger_file_level=logging.DEBUG
                 ):
        super().__init__(queue_sender, True, to_tensor, device)
        self.episodes = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.priority_replay = priority_replay
        self.num_cached = 0

        batch_maker = SelectBatcher(self.to_tensor, self.device)
        self.batch_maker = connection.MultiProcessJobExecutors(batch_maker,
                                                               send_generator=self.send_raw_batch(),
                                                               num=num_batch_maker,
                                                               queue_receiver=self.queue_sender,
                                                               name_prefix="batch_maker",
                                                               logger_file_dir=logger_file_dir,
                                                               file_level=logger_file_level)

    def cache(self, episodes: List[Tuple[int, List[bytes]]]):
        self.episodes.extend(episodes)
        self.num_cached += len(episodes)
        logging.debug(f"total cached episodes is {self.num_cached}")

    def send_raw_batch(self):
        while True:
            yield [self.send_a_sample() for _ in range(self.batch_size)]

    def send_a_sample(self):
        while len(self.episodes) == 0:
            pass

        if self.priority_replay:
            while True:
                ep_idx = random.randrange(0, len(self.episodes))
                accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.episodes.maxlen
                if random.random() < accept_rate:
                    index, ep = self.episodes[ep_idx]
                    st = random.randrange(0, len(ep) - 1)  # change start turn by sequence length
                    return index, ep[st], ep[st+1]
        else:
            ep_idx = random.randrange(0, len(self.episodes))
            index, ep = self.episodes[ep_idx]
            st = random.randrange(0, len(ep) - 1)  # change start turn by sequence length
            return index, ep[st], ep[st + 1]

    def start(self):
        self.batch_maker.start()


class MemoryServer:
    def __init__(self,
                 memory_replay: MemoryReplayBase,
                 port: int,
                 actor_num=None,
                 ):

        self.actor_communicator = connection.QueueCommunicator(port, actor_num)
        self.memory_replay = memory_replay
        self.actor_num = actor_num

    def run(self):
        logging.info("start server to receive episodes that generated by actor")
        self.actor_communicator.run()
        self.memory_replay.start()
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            """
            cmd: Literal["episodes"]
            """
            logging.debug(cmd)
            self.memory_replay.cache(data)
            self.actor_communicator.send(conn, (cmd, "successfully receive episodes"))

    def run_sync(self):
        self.actor_communicator.run_sync()

        conns = []
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            """
            cmd: Literal["episodes"]
            """
            logging.debug(cmd)

            self.memory_replay.cache(data)
            conns.append(conn)

            if len(conns) == self.actor_num:
                self.memory_replay.start()
                for conn in conns:
                    self.actor_communicator.send(conn, (cmd, "successfully receive episodes"))
                conns.clear()







