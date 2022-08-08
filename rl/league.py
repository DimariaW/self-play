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

from typing import Dict, Callable, Any, List
from collections import defaultdict


class League:
    def __init__(self,
                 queue_receiver: connection.Receiver,
                 port: int,
                 num_actors: int = None,
                 use_bz2: bool = True,
                 cache_weights_intervals: int = 1,
                 save_weights_dir: str = None,
                 save_weights_intervals: int = 100000000,
                 tensorboard_dir: str = None,
                 ):
        """
        1. 从learner端接收模型队列
        """
        self.queue_receiver = queue_receiver
        """
        2. 与actor通信模块
        """
        self.port = port
        self.num_actors = num_actors
        self.actor_communicator = connection.QueueCommunicator(port, num_client=self.num_actors)
        """
        3. receive shared model and shared model index (num_update)
           model_id = (model_name, model_index)  # 唯一标识符
        """
        self.num_update, self.model = self.queue_receiver.recv()
        self.num_update: torch.Tensor
        self.model: model.Model
        self.model_name = self.model.name
        logging.info(f"successfully received shared memory model: {self.model_name}")
        """
        4. cache latest model weights
        """
        self.use_bz2 = use_bz2
        weights = self.model.get_weights()
        if self.use_bz2:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        else:
            self.cached_weights = weights
        self.cached_weights_index = self.num_update.item()
        self.cache_weights_intervals = cache_weights_intervals
        """
        5. save model at a fix interval, used as opponent to self-play
        """
        if save_weights_dir is not None:
            self.save_weights_dir = save_weights_dir
            os.makedirs(self.save_weights_dir, exist_ok=True)

        self.saved_weights = self.cached_weights
        self.saved_weights_index = self.cached_weights_index
        self.save_weights_intervals = save_weights_intervals
        self.self_play_winning_rate = 0.5
        """
        6. initialize opponents_pool, (model_name, model_index) : model_weights
           initialize opponents winning, (model_name, model_index) : winning_rate    # 当前cached weights 对每个对手的winning rate
        """
        self.opponents_pool = {}  # exclude saved weights which is used to self-play
        self.win_prob_to_opponents = {}
        """
        7. use tensorboard to log sample infos and eval infos
        """
        if tensorboard_dir is not None:
            self.num_received_infos = defaultdict(lambda: 0)
            self.sw = tensorboardX.SummaryWriter(logdir=tensorboard_dir)
        """
        8. create response functions
        """

    def add_opponent(self, name: str, index: int = None, weights=None, winning_rate=0.5):
        self.opponents_pool[(name, index)] = weights
        self.win_prob_to_opponents[(name, index)] = winning_rate
        logging.debug(f"current win prob is {self.win_prob_to_opponents}")

    @utils.wrap_traceback
    def _update(self):
        while True:
            self._update_once()

    def _update_once(self):
        """
        1. 更新cached weights
        """
        if self.num_update.item() - self.cached_weights_index >= self.cache_weights_intervals:
            self._update_cached_weights()
            """ 
            2. 如果需要save model weights, 则只在固定时间步之后更新save, or update saved_weights to cached_weights
            """
            if self.cached_weights_index - self.saved_weights_index >= self.save_weights_intervals:
                self._update_saved_weights()

    def _update_cached_weights(self):
        self.cached_weights_index = self.num_update.item()
        weights = self.model.get_weights()
        if not self.use_bz2:
            self.cached_weights = weights
        else:
            self.cached_weights = bz2.compress(pickle.dumps(weights))
        logging.debug(f"successfully update model to model id {self.cached_weights_index}")

    def _update_saved_weights(self):
        # 将当前的saved weights 加入对手池
        self.add_opponent(self.model_name, self.saved_weights_index, self.saved_weights, self.self_play_winning_rate)

        self.saved_weights_index = self.cached_weights_index
        self.saved_weights = self.cached_weights
        self.self_play_winning_rate = 0.5

        if hasattr(self, "save_weights_dir"):
            if self.use_bz2:
                weights = pickle.loads(bz2.decompress(self.saved_weights))
            else:
                weights = self.saved_weights

            pickle.dump(weights,
                        open(
                            osp.join(self.save_weights_dir, f"{self.model_name}_{self.saved_weights_index}.pickle"),
                            "wb")
                        )

    def run(self):
        create_response_functions: Dict[str, Callable] = {
            "model": self._send_model,
            "sample_infos": self._record_sample_infos,
            "eval_infos": self._record_eval_infos
        }
        threading.Thread(target=self._update, args=(), daemon=True).start()
        self.actor_communicator.run()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            create_response_functions[cmd](conn, cmd, data)
            """
            cmd: Literal["model", "eval_infos", "sample_infos"]
            
            cmd          data              explanation
            model        latest            request cached weights by the sample actor
            model        sample_opponent   request opponents by sample actor                                  
            model        -1                request saved weights by the eval actor
            model        eval_opponent     request opponents by eval actor
            
            sample_infos    dict           data must contain "win" key, "model_id", "opponent_id"
            eval_infos      dict           data must contain "win" key, "model_id", "opponent_id"
            """

    def _send_model(self, conn: connection.PickledConnection, cmd: str, data: str):
        logging.debug(f"cmd: {cmd}, data: {data}")

        if data == "latest":
            self._send_cached_weights(conn, cmd)

        elif data == "sample_opponent":
            """
            80 % self play, 20% pfsp
            """
            ratio = np.random.rand()
            if ratio <= 0.8:
                self._send_saved_weights(conn, cmd)
            else:
                self._send_opponent(conn, cmd)

        elif data == -1:
            self._send_saved_weights(conn, cmd)

        elif data == "eval_opponent":
            self._send_opponent(conn, cmd, ignore_priority=True)

    def _send_cached_weights(self, conn: connection.PickledConnection, cmd: str):
        model_id = (self.model_name, self.cached_weights_index)
        weights = self.cached_weights
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))

    def _send_saved_weights(self, conn: connection.PickledConnection, cmd: str):
        model_id = (self.model_name, self.saved_weights_index)
        weights = self.saved_weights
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))

    def _send_opponent(self, conn: connection.PickledConnection, cmd: str, ignore_priority=False):
        def normalize(winning_rate: tuple, p=1):
            logits = (1.-np.array(winning_rate))**p
            return logits/np.sum(logits)

        opponents_id, opponents_winning_rate = tuple(zip(*self.win_prob_to_opponents.items()))
        if ignore_priority:
            model_id = random.choice(opponents_id)
        else:
            model_id = random.choices(opponents_id, weights=normalize(opponents_winning_rate), k=1)[0]
        weights = self.opponents_pool[model_id]
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))

    def _record_sample_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        """
        assert cmd == "sample_infos"
        sample_info have model_id, opponent_id, win and other scalar values.
        """
        logging.debug(f"{cmd}: {data}")

        for sample_info in data:
            # update winning rate
            opponent_id = sample_info["opponent_id"]
            if opponent_id == (self.model_name, self.saved_weights_index):
                self.self_play_winning_rate += 0.001 * (sample_info["win"] - self.self_play_winning_rate)
            else:
                win_prob = self.win_prob_to_opponents[opponent_id]
                self.win_prob_to_opponents[opponent_id] += 0.001 * (sample_info["win"] - win_prob)
            # log to tensorboard
            if hasattr(self, "sw"):
                self._record_info(sample_info, suffix="sample")

        self.actor_communicator.send(conn, (cmd, "successfully receive and record sample_infos"))

    def _record_eval_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        logging.debug(f"{cmd}: {data}")

        # log to tensorboard
        if hasattr(self, "sw"):
            for eval_info in data:
                self._record_info(eval_info, suffix="eval")

        self.actor_communicator.send(conn, (cmd, "successfully receive and record eval_infos"))

    def _record_info(self, info: Dict[str, Any], suffix: str):
        opponent_id = info["opponent_id"]
        tag = f'vs_{opponent_id[0]}_{opponent_id[1]}_{suffix}'
        self.num_received_infos[tag] += 1
        for key, value in info.items():
            if key == "opponent_id":
                continue
            if key == "model_id":
                model_name, model_index = value
                self.sw.add_scalar(tag=f"{tag}/{model_name}_index",
                                   scalar_value=model_index,
                                   global_step=self.num_received_infos[tag])
                continue
            self.sw.add_scalar(tag=f"{tag}/{key}",
                                   scalar_value=value,
                                   global_step=self.num_received_infos[tag])







