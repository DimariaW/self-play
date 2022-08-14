from tests.football.actor import ActorMain
import rl.actor as rl_actor
import rl.utils as utils
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle
import bz2

utils.set_process_logger()
env, agent = ActorMain().create_env_and_agent()
env.render()

actor = rl_actor.Actor(env, agent, num_episodes=10)

weights = pickle.load(open("./1_vs_1_model/cnn_132000.pickle", "rb"))
index = 132000

actor.reset_agent(("cnn", index), weights)
actor.reset_env(("builtin_ai", None), None)

for _, data in actor.predict():
    logging.info(data)
    actor.reset_env(("builtin_ai", None), None)

class League(ModelServer4RecordAndEval):
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
        self.self_play_winning_rate = 0.5
        # exclude saved weights which is used to self-play
        self.opponents_pool: Dict[Tuple[str, Optional[int]], Optional[Mapping[str, np.ndarray], bytes]] = {}
        self.win_prob_to_opponents: Dict[Tuple[str, Optional[int]], float] = {}

    def add_opponent(self, name: str, index: Optional[int] = None,
                     weights: Optional[Mapping[str, np.ndarray], bytes] = None,
                     winning_rate: float = 0.):
        if weights is not None:
            if not isinstance(weights, bytes) and self.use_bz2:
                weights = bz2.compress(pickle.dumps(weights))
            elif isinstance(weights, bytes) and not self.use_bz2:
                weights = pickle.loads(bz2.decompress(weights))

        self.opponents_pool[(name, index)] = weights
        self.win_prob_to_opponents[(name, index)] = winning_rate
        logging.debug(f"current win prob is {self.win_prob_to_opponents}")

    def check_to_update(self):
        _ = self.win_prob_to_opponents
        _ = self.self_play_winning_rate
        return False

    def _update_once(self):
        """
        1. 更新cached weights
        """
        if self.index.item() - self.cached_weights_index >= self.cache_weights_intervals or self.check_to_update():
            self._update_cached_weights()

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

    def _send_saved_weights(self, conn: connection.PickledConnection, cmd: str):
        model_id = (self.model_name, self.saved_weights_index)
        weights = self.saved_weights
        self.actor_communicator.send(conn, (cmd, (model_id, weights)))

    def _send_opponent(self, conn: connection.PickledConnection, cmd: str, ignore_priority=False):
        def normalize(winning_rate: tuple, p=1):
            logits = (1. - np.array(winning_rate)) ** p
            return logits / np.sum(logits)

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
