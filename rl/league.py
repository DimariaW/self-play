import rl.connection as connection
import rl.utils as utils
import rl.model as model

import tensorboardX
import threading
import numpy as np
import os
import os.path as osp
import logging
import bz2
import pickle
import torch
import random

from typing import Dict, Callable, Any, List, Tuple, Optional, Mapping
from collections import defaultdict


class ModelServerBase:
    def __init__(self, queue_receiver: connection.Receiver,
                 port: int,
                 num_actors: int = None,
                 use_bz2: bool = True,
                 ):
        # 1. 从learner端接收模型队列
        self.queue_receiver = queue_receiver
        # 2. 与actor通信模块
        self.port = port
        self.num_actors = num_actors
        self.actor_communicator = connection.QueueCommunicator(port, num_client=self.num_actors)
        # 3. 得到共享模型
        self.index, self.model = self.queue_receiver.recv()
        self.index: torch.Tensor
        self.model: model.Model
        self.model_name = self.model.name
        logging.info(f"successfully received shared memory model: {self.model_name}")
        self.use_bz2 = use_bz2

    @utils.wrap_traceback
    def _update(self):
        while True:
            self._update_once()

    def _update_once(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class ModelServer(ModelServerBase):
    """
    used to update the latest model, sampler_actor should request the latest model from this server
    """
    def __init__(self, queue_receiver: connection.Receiver,
                 port: int,
                 num_actors: int = None,
                 use_bz2: bool = True,
                 cache_weights_intervals: int = 1):
        super().__init__(queue_receiver, port, num_actors, use_bz2)
        weights = self.model.get_weights()
        if self.use_bz2:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        else:
            self.cached_weights = weights
        self.cached_weights_index = self.index.item()
        self.cache_weights_intervals = cache_weights_intervals

    def _update_cached_weights(self):
        self.cached_weights_index = self.index.item()
        weights = self.model.get_weights()
        if not self.use_bz2:
            self.cached_weights = weights
        else:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        logging.debug(f"successfully update model to model id {self.cached_weights_index}")

    def _update_once(self):
        if self.index.item() - self.cached_weights_index >= self.cache_weights_intervals:
            self._update_cached_weights()

    def run(self):
        response_function = {
            "model": self._send_cached_weights
        }

        threading.Thread(target=self._update, args=(), daemon=True).start()
        self.actor_communicator.run()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            response_function[cmd](conn, cmd, data)

    def _send_cached_weights(self, conn: connection.PickledConnection, cmd: str, data: str):
        model_id = (self.model_name, self.cached_weights_index)
        weights = self.cached_weights
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))
        logging.debug(f"receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {model_id})")


class ModelServer4RecordAndEval(ModelServer):
    def __init__(self, queue_receiver: connection.Receiver,
                 port: int,
                 num_actors: int = None,
                 use_bz2: bool = True,
                 cache_weights_intervals: int = 3000,
                 save_weights_dir: str = None,
                 tensorboard_dir: str = None
                 ):
        """
        used for save model weights to disk and evaluation
        """
        super().__init__(queue_receiver, port, num_actors, use_bz2, cache_weights_intervals)

        if save_weights_dir is not None:
            self.save_weights_dir = save_weights_dir
            os.makedirs(self.save_weights_dir, exist_ok=True)

        if tensorboard_dir is not None:
            self.num_received_infos = defaultdict(int)
            self.sw = tensorboardX.SummaryWriter(logdir=tensorboard_dir)

    def _update_cached_weights(self):
        super()._update_cached_weights()
        if hasattr(self, "save_weights_dir"):
            if self.use_bz2:
                weights = pickle.loads(bz2.decompress(self.cached_weights))
            else:
                weights = self.cached_weights

            pickle.dump(weights,
                        open(
                            osp.join(self.save_weights_dir, f"{self.model_name}_{self.cached_weights_index}.pickle"),
                            "wb")
                        )

    def _record_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        for info in data:
            self._record_info(info, cmd)
        self.actor_communicator.send(conn, (cmd, "successfully send infos"))
        logging.debug(f"receive cmd: {cmd}, send (cmd: {cmd}, data: successfully send infos)")

    def _record_info(self, info: Dict[str, Any], tag):
        self.num_received_infos[tag] += 1
        for key, value in info.items():
            if key != "meta_info":
                self.sw.add_scalar(tag=f"{tag}/{key}",
                                   scalar_value=value,
                                   global_step=self.num_received_infos[tag])

    def run(self):
        response_function = {
            "model": self._send_cached_weights,
            "sample_infos": self._record_infos,
            "eval_infos": self._record_infos
        }
        threading.Thread(target=self._update, args=(), daemon=True).start()
        self.actor_communicator.run()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            response_function[cmd](conn, cmd, data)







