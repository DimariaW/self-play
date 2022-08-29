import torch
import multiprocessing as mp
import os

import rl.core as core
import rl.memory as mem
import rl.algorithm as alg
import rl.league as lg

from tests.env_models import cartpole
from tests.impala.config import CONFIG, USE_BZ2


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        traj_queue = mem.TrajQueueMP(maxlen=8,
                                     queue_sender=queue_sender,
                                     batch_size=16,
                                     use_bz2=USE_BZ2,
                                     to_tensor=True,
                                     device=device,
                                     # batch_maker args
                                     num_batch_maker=1,
                                     logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                     logger_file_level=self.logger_file_level)

        memory_server = mem.MemoryServer(traj_queue, self.port, actor_num=None)
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor_receiver = self.create_receiver(queue_receiver)
        model = CONFIG["model_class"](**CONFIG["model_args"]).to(device)
        # model = Model(CONFIG["obs_dim"], CONFIG["num_act"], use_orthogonal_init=True, use_tanh=False).to(device)
        # model = ModelLSTM(CONFIG["obs_dim"], CONFIG["num_act"]).to(device)
        impala = alg.IMPALA(model, queue_senders, tensor_receiver,
                            lr=2e-4, gamma=0.99, lbd=0.98, vf=0.5, ef=1e-4,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "learn_info"),
                            vtrace_key=["reward"],
                            # only_critic=["reward"],
                            # upgo_key=["reward"]
                            )
        impala.run()


class ModelServerMain(core.ModelServerMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        model_server = lg.ModelServer(queue_receiver, self.port, use_bz2=USE_BZ2, cache_weights_intervals=1)
        model_server.run()


class LeagueMain(core.ModelServerMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        """
        league = lg.ModelServer4RecordAndEval(queue_receiver, self.port, use_bz2=USE_BZ2,
                                              cache_weights_intervals=1000,
                                              save_weights_dir=os.path.join(self.logger_file_dir, "models"),
                                              tensorboard_dir=os.path.join(self.logger_file_dir, "metrics"))
        """
        league = lg.League(queue_receiver, self.port, use_bz2=USE_BZ2,
                           cache_weights_intervals=1000,
                           save_weights_dir=os.path.join(self.logger_file_dir, "models"),
                           tensorboard_dir=os.path.join(self.logger_file_dir, "metrics"))
        league.run()
