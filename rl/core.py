import os
import logging
import multiprocessing as mp
import functools
import torch
from typing import List, Tuple, Union, Literal

import rl.utils as utils
from rl.connection import Receiver


from rl.actor import ActorMainBase, open_gather

__all__ = ["ActorMainBase", "MemoryMainBase", "LearnerMainBase", "ModelServerMainBase", "open_gather", "train_main"]


class MainBase:
    def __init__(self, name, logger_file_dir, logger_file_level):
        """
        set logger file path to "logger_file_dir/name.txt"
        :param name: the name of file
        :param logger_file_dir: the logger file directory
        """
        self.logger_file_dir = logger_file_dir
        self.logger_file_level = logger_file_level
        self.logger_file_path = None
        if self.logger_file_dir is not None:
            self.logger_file_path = os.path.join(self.logger_file_dir, f"{name}.txt")


class MemoryMainBase(MainBase):
    def __init__(self, port: int, logger_file_dir=None, logger_file_level=logging.DEBUG):
        super().__init__("memory", logger_file_dir, logger_file_level)
        self.port = port

    def __call__(self, queue_sender: mp.Queue):
        utils.set_process_logger(file_path=self.logger_file_path, file_level=self.logger_file_level)
        utils.wrap_traceback(self.main)(queue_sender)

    def main(self, queue_sender: mp.Queue):
        """
        自此函数中实例化MemoryServer对象，处理actor收集的数据.
        """
        raise NotImplementedError


class LearnerMainBase(MainBase):
    def __init__(self, logger_file_dir=None, logger_file_level=logging.DEBUG):
        super().__init__("learner", logger_file_dir, logger_file_level)

    def __call__(self, queue_receiver: mp.Queue, queue_senders: Union[List[mp.Queue], Tuple[mp.Queue]]):
        utils.set_process_logger(file_path=self.logger_file_path, file_level=self.logger_file_level)
        utils.wrap_traceback(self.main)(queue_receiver, queue_senders)

    def main(self, queue_receiver: mp.Queue, qqueue_senders: Union[List[mp.Queue], Tuple[mp.Queue]]):
        """
        在此函数中实例化myrl.Algorithm的子类， 并且调用run 函数运行.
        """
        raise NotImplementedError

    @staticmethod
    def create_receiver(queue_receiver: mp.Queue, to_tensor=False, device=torch.device("cpu"), num_sender=-1):
        if not to_tensor:
            return Receiver(queue_receiver, num_sender, postprocess=None)

        to_device = functools.partial(utils.to_tensor, device=device, unsqueeze=None)
        return Receiver(queue_receiver, num_sender, postprocess=lambda data: (data[0], to_device(data[1])))


#%%
class _ModelMainBase(MainBase):
    def __call__(self, queue_receiver: mp.Queue):
        utils.set_process_logger(file_path=self.logger_file_path, file_level=self.logger_file_level)
        utils.wrap_traceback(self.main)(queue_receiver)

    def main(self, queue_receiver: mp.Queue):
        """
        the process to manage model weights.
        """
        raise NotImplementedError

    @staticmethod
    def create_receiver(queue_receiver: mp.Queue, num_sender=-1):
        return Receiver(queue_receiver, num_sender=num_sender, postprocess=None)


class ModelMainBase(_ModelMainBase):
    def __init__(self, port: int, logger_file_dir=None, logger_file_level=logging.DEBUG):
        super().__init__("model", logger_file_dir, logger_file_level)
        self.port = port

    def main(self, queue_receiver: mp.Queue):
        """
        the process to manage model weights.
        """
        raise NotImplementedError


class LeagueMainBase(_ModelMainBase):
    def __init__(self, port: int, logger_file_dir=None, logger_file_level=logging.DEBUG):
        super().__init__("league", logger_file_dir, logger_file_level)
        self.port = port

    def main(self, queue_receiver: mp.Queue):
        """
        the process to manage model weights.
        """
        raise NotImplementedError


def train_main(learner_main: LearnerMainBase,
               memory_mains: Union[List[MemoryMainBase], Tuple[MemoryMainBase]],
               model_league_mains: Union[List[Union[ModelMainBase, LeagueMainBase]],
                                         Tuple[Union[ModelMainBase, LeagueMainBase]]],
               memory_buffer_length=1):  # receiver and sender]],

    mp.set_start_method("spawn")
    """
    1. learner
    """
    queue_receiver = mp.Queue(maxsize=memory_buffer_length)
    # receiver batched tensor, when on policy, this can be set to 1
    queue_senders = []
    for _ in range(len(model_league_mains)):
        queue_senders.append(mp.Queue(maxsize=1))  # the queue to send the newest data
    learner_process = mp.Process(target=learner_main, args=(queue_receiver, queue_senders),
                                 daemon=False, name="learner_main")
    learner_process.start()
    """
    model process
    """
    model_processes = []
    for i, model_main in enumerate(model_league_mains):
        model_process = mp.Process(target=model_main, args=(queue_senders[i],),
                                   daemon=False, name=f"model_league_{i}")
        model_process.start()
        model_processes.append(model_process)
    """
    memory process
    """
    memory_processes = []
    for i, memory_main in enumerate(memory_mains):
        memory_process = mp.Process(target=memory_main, args=(queue_receiver,),
                                    daemon=False, name=f"memory_main_{i}")
        memory_process.start()
        memory_processes.append(memory_process)
    """
    cleaning when done
    """
    try:
        learner_process.join()
        for model_process in model_processes:
            model_process.join()
        for memory_process in memory_processes:
            memory_process.join()
    finally:
        learner_process.close()
        for model_process in model_processes:
            model_process.close()
        for memory_process in memory_processes:
            memory_process.close()








