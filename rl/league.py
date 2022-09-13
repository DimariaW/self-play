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

from typing import Dict, Callable, Any, List, Tuple, Optional, Mapping, Union
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
        logging.debug(f"successfully update model to index: {self.cached_weights_index}")

    def _update_once(self):
        if self.index.item() - self.cached_weights_index >= self.cache_weights_intervals:
            self._update_cached_weights()

    def run(self):
        response_function = {
            "model": self._run_model
        }

        threading.Thread(target=self._update, args=(), daemon=True).start()
        self.actor_communicator.run()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            response_function[cmd](conn, cmd, data)

    def _run_model(self, conn: connection.PickledConnection, cmd: str, data: str):
        model_id = (self.model_name, self.cached_weights_index)
        weights = self.cached_weights
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))
        logging.debug(f"receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {model_id})")


class ModelServer4Evaluation(ModelServer):
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
            self.tag2steps = defaultdict(int)
            self.sw = tensorboardX.SummaryWriter(logdir=tensorboard_dir)

    def _save_cached_weights(self):
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

    def _update_once(self):
        if self.index.item() - self.cached_weights_index >= self.cache_weights_intervals:
            self._update_cached_weights()
            self._save_cached_weights()

    def run(self):
        response_function = {
            "model": self._run_model,
            "sample_infos": self._run_sample_infos,
            "eval_infos": self._run_eval_infos
        }
        threading.Thread(target=self._update, args=(), daemon=True).start()
        self.actor_communicator.run()

        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            response_function[cmd](conn, cmd, data)

    def _run_sample_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        """assert cmd == "sample_infos
        """
        for info in data:
            self._record_info(info, cmd)
        self.actor_communicator.send(conn, (cmd, "successfully receive sample_infos by server"))
        logging.debug(f"receive (cmd: {cmd}), send (cmd: {cmd}, data: successfully receive sample_infos by server)")

    def _run_eval_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        """assert cmd == "eval_infos"
        """
        for info in data:
            self._record_info(info, cmd)
        self.actor_communicator.send(conn, (cmd, "successfully receive eval_infos by server"))
        logging.debug(f"receive (cmd: {cmd}), send (cmd: {cmd}, data: successfully receive eval_infos by server)")

    def _record_info(self, info: Dict[str, Any], tag):
        self.tag2steps[tag] += 1
        for key, value in info.items():
            if key != "meta_info":
                self.sw.add_scalar(tag=f"{tag}/{key}",
                                   scalar_value=value,
                                   global_step=self.tag2steps[tag])


class League(ModelServer4Evaluation):
    """
    sample opponent: cached weights and opponent pool
    eval moodel: cached weights
    eval opponent: opponent pool
    """
    def __init__(self,
                 queue_receiver: connection.Receiver,
                 port: int,
                 num_actors: int = None,
                 use_bz2: bool = True,
                 cache_weights_intervals: int = 3000,
                 save_weights_dir: str = None,
                 tensorboard_dir: str = None,
                 ):
        super().__init__(queue_receiver, port, num_actors, use_bz2,
                         cache_weights_intervals, save_weights_dir, tensorboard_dir)
        """
        initialize opponents_pool, (model_name, model_index) : model_weights
        initialize opponents winning, (model_name, model_index) : winning_rate    # 当前cached weights 对每个对手的winning rate
        """
        self.self_play_winning_rate = 0.
        # exclude saved weights which is used to self-play
        self.opponents_pool: Dict[Tuple[str, Optional[int]], Optional[Mapping[str, np.ndarray], bytes]] = {}
        self.win_rate_to_opponents: Dict[Tuple[str, Optional[int]], float] = {}

        self.add_opponent(self.model_name, self.cached_weights_index-1,
                          weights=self.cached_weights,
                          winning_rate=self.self_play_winning_rate)

    def add_opponent(self, name: str, index: Optional[int] = None,
                     weights: Optional[Union[Mapping[str, np.ndarray], bytes]] = None,
                     winning_rate: float = 0.):

        if weights is not None:
            if not isinstance(weights, bytes) and self.use_bz2:
                weights = bz2.compress(pickle.dumps(weights))
            elif isinstance(weights, bytes) and not self.use_bz2:
                weights = pickle.loads(bz2.decompress(weights))

        self.opponents_pool[(name, index)] = weights
        self.win_rate_to_opponents[(name, index)] = winning_rate
        logging.debug(f"current win rate is {self.win_rate_to_opponents}")

    def check_to_update(self):
        mean_win_rate = np.mean([win_rate for _, win_rate in self.win_rate_to_opponents.items()])
        return self.self_play_winning_rate >= 0.7 and mean_win_rate >= 0.7

    def _update_once(self):
        """
        1. 更新cached weights
        """
        if self.index.item() - self.cached_weights_index >= self.cache_weights_intervals or self.check_to_update():
            self.add_opponent(self.model_name, self.cached_weights_index,
                              weights=self.cached_weights, winning_rate=self.self_play_winning_rate)
            self._update_cached_weights()
            self.self_play_winning_rate = 0.
            self._save_cached_weights()

    def run(self):
        create_response_functions: Dict[str, Callable] = {
            "model": self._send_model,
            "sample_infos": self._record_sample_infos,
            "eval_infos": self._run_sample_infos
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
            model        eval_opponent     request opponents by eval actor

            sample_infos    dict           data must contain "win" key, "model_id", "opponent_id"
            eval_infos      dict           data must contain "win" key, "model_id", "opponent_id"
            """

    def _send_model(self, conn: connection.PickledConnection, cmd: str, data: str):
        logging.debug(f"cmd: {cmd}, data: {data}")

        if data == "latest":
            self._run_model(conn, cmd, data)

        elif data == "sample_opponent":
            """
            50 % self play, 50 % pfsp
            """
            ratio = np.random.rand()
            if ratio <= 0.5:
                self._run_model(conn, cmd, data)
            elif ratio <= 0.55:
                self._send_opponent(conn, cmd, data, ignore_priority=True)
            else:  # pfsp
                self._send_opponent(conn, cmd, data)

        elif data == "eval_opponent":
            self._send_opponent(conn, cmd, data, ignore_priority=True)

    def _send_opponent(self, conn: connection.PickledConnection, cmd: str, data: str, ignore_priority=False):
        def normalize(winning_rate: tuple, p=1):
            logits = (1. - np.array(winning_rate) + 1e-3) ** p
            return logits / np.sum(logits)

        opponents_id, opponents_winning_rate = tuple(zip(*self.win_rate_to_opponents.items()))
        if ignore_priority:
            model_id = random.choice(opponents_id)
        else:
            model_id = random.choices(opponents_id, weights=normalize(opponents_winning_rate), k=1)[0]
        weights = self.opponents_pool[model_id]
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))
        logging.debug(f"receive (cmd: {cmd}, data: {data}), send (cmd: {cmd}, data: {model_id})")

    def _record_sample_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        """
        assert cmd == "sample_infos"
        sample_info have model_id, opponent_id, win and other scalar values.
        """
        logging.debug(f"{cmd}: {data}")

        for info in data:
            # update winning rate
            opponent_id = info["meta_info"]["opponent_id"]
            if opponent_id == (self.model_name, self.cached_weights_index):
                self.self_play_winning_rate += 0.001 * (info["meta_info"]["win"] - self.self_play_winning_rate)
            else:
                win_rate = self.win_rate_to_opponents[opponent_id]
                self.win_rate_to_opponents[opponent_id] += 0.001 * (info["meta_info"]["win"] - win_rate)
            # log to tensorboard
        self._run_sample_infos(conn, cmd, data)

    def _run_sample_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        for info in data:
            agent_name, agent_index = info["meta_info"]["agent_id"]
            opponent_name, opponent_index = info["meta_info"]["opponent_id"]

            if opponent_index is not None:
                tag = f"{cmd}_{agent_name}_vs_{opponent_name}{opponent_index}"
            else:
                tag = f"{cmd}_{agent_name}_vs_{opponent_name}"

            self._record_info(info, tag=tag)

        self.actor_communicator.send(conn, (cmd, "successfully send infos"))
        logging.debug(f"receive cmd: {cmd}, send (cmd: {cmd}, data: successfully send infos)")


class ELOEvaluator:
    """
    model_pools 维护所有的model, 每个model唯一标识符为(name, index)， 属性为weights, elo score
    每个model至少对战 8*ceil(log_2^N)+1局
    任意两个model的预估胜率为：
    P(A win B) = 1/(1 + 10^(-D/400)), D为A_score - B_score, (分差400， 胜率超90%)
    更新规则 A_score += 32*( 1/0/0.5 - P(A win B) )
    """
    def __init__(self, port: int,
                 num_actors: int = None,
                 use_bz2: bool = True,
                 tensorboard_dir: str = None):
        self.actor_communicator = connection.QueueCommunicator(port, num_client=num_actors)
        self.use_bz2 = use_bz2

        self.model_ids = []
        self.model_id_to_weights = {}
        self.model_id_to_elo_score = {}
        self.model_id_to_game_times = {}

        if tensorboard_dir is not None:
            self.num_received_infos = defaultdict(int)
            self.sw = tensorboardX.SummaryWriter(logdir=tensorboard_dir)

    def add_model(self, model_name: str, model_index: int = None, weights=None, elo_score: float = 1200):
        model_id = model_name, model_index
        assert model_id not in self.model_ids

        if weights is not None:
            if not isinstance(weights, bytes) and self.use_bz2:
                weights = bz2.compress(pickle.dumps(weights))
            elif isinstance(weights, bytes) and not self.use_bz2:
                weights = pickle.loads(bz2.decompress(weights))

        self.model_ids.append(model_id)
        self.model_id_to_weights[model_id] = weights
        self.model_id_to_elo_score[model_id] = elo_score
        self.model_id_to_game_times[model_id] = 0

    def sample_models(self):
        model_id, opponent_id = self._sample_models()
        self.model_id_to_game_times[model_id] += 1
        self.model_id_to_game_times[opponent_id] += 1
        return ((model_id, self.model_id_to_weights[model_id]),
                (opponent_id, self.model_id_to_weights[opponent_id]))

    def _sample_models(self):
        selected_id = min(self.model_ids, key=lambda model_id: self.model_id_to_game_times[model_id])
        selected_elo_score = self.model_id_to_elo_score[selected_id]
        other_model_id_elo_scores = [(model_id, self.model_id_to_elo_score[model_id])
                                     for model_id in self.model_ids if model_id != selected_id]
        other_model_ids, other_elo_scores = list(zip(*other_model_id_elo_scores))
        opponent_id = other_model_ids[self._sample_opponent_index(selected_elo_score, np.array(list(other_elo_scores)))]
        return selected_id, opponent_id

    def _sample_opponent_index(self, selected_elo_score: float, other_elo_scores: np.ndarray):
        elo_score_diffs = selected_elo_score - other_elo_scores
        win_probs = self._cal_win_probs(elo_score_diffs)
        sample_probs = (1-win_probs) * win_probs
        return random.choices(np.arange(len(other_elo_scores)), weights=sample_probs, k=1)[0]

    @staticmethod
    def _cal_win_probs(elo_score_diffs):
        return 1/(1 + np.power(10, -elo_score_diffs/400))

    def update_elo_score(self, info):
        model_id = info["model_id"]
        opponent_id = info["opponent_id"]
        win = info["win"]

        prior_win_prob = self._cal_win_probs(self.model_id_to_elo_score[model_id]
                                             - self.model_id_to_elo_score[opponent_id])
        self.model_id_to_elo_score[model_id] += 32 * (win - prior_win_prob)
        self.model_id_to_elo_score[opponent_id] += 32 * (prior_win_prob - win)

        if hasattr(self, "sw"):
            model_tag = model_id[0] if model_id[1] is None else f"{model_id[0]}_{model_id[1]}"
            opponent_tag = model_id[0] if model_id[1] is None else f"{model_id[0]}_{model_id[1]}"
            # win tag
            self._log_to_tensorboard(f"win/{model_tag}_vs_{opponent_tag}", win)
            # elo_score
            self._log_to_tensorboard(f"elo_score/{model_tag}", self.model_id_to_elo_score[model_id])
            self._log_to_tensorboard(f"elo_score/{opponent_tag}", self.model_id_to_elo_score[opponent_id])
            return model_tag, opponent_tag

    def _log_to_tensorboard(self, tag, value):
        self.num_received_infos[tag] += 1
        self.sw.add_scalar(tag, value, global_step=self.num_received_infos[tag])

    def log_elo_score(self):
        model_id_and_elo_score = [(model_id, elo_score) for model_id, elo_score in self.model_id_to_elo_score.items()]
        model_id_and_elo_score_sorted = sorted(model_id_and_elo_score, key=lambda x: -x[1])
        logging.info(model_id_and_elo_score_sorted)

    def run(self):
        cmd2func = {
            "model": self._run_model,
            "eval_infos": self._run_eval_infos
        }
        self.actor_communicator.run()
        while True:
            conn, (cmd, data) = self.actor_communicator.recv()
            """
            cmd: model, data: eval_model_and_opponent
            cmd: eval_infos, data: list[info]
            """
            cmd2func[cmd](conn, cmd, data)

    def _run_model(self, conn, cmd, data):
        (model_id, model_weights), (opponent_id, opponent_weights) = self.sample_models()
        self.actor_communicator.send(conn, (cmd, ((model_id, model_weights), (opponent_id, opponent_weights))))
        logging.debug(f"recv cmd:{cmd}, data:{data}, send cmd:{cmd}, data:{model_id}, {opponent_id}")

    def _run_eval_infos(self, conn: connection.PickledConnection, cmd: str, data: List[Dict[str, Any]]):
        for info in data:
            meta_info = info["meta_info"]
            if hasattr(self, "sw"):
                model_tag, opponent_tag = self.update_elo_score(meta_info)
                for key, value in info.items():
                    if key != "meta_info":
                        self._log_to_tensorboard(f"{key}/{model_tag}_vs_{opponent_tag}", value)
            else:
                self.update_elo_score(meta_info)
        self.actor_communicator.send(conn, (cmd, "successfully received data by server!"))
        logging.debug(f"recv cmd:{cmd}, data:{data}, send cmd:{cmd}, data: successfully received data by server!")
        self.log_elo_score()
