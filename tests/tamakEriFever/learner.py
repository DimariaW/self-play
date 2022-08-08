
import torch
import multiprocessing as mp
import os
import rl.core as core
import rl.memory as mem
import rl.algorithm as alg
import rl.league as lg
import numpy as np
import logging

from tests.tamakEriFever.football_model import FootballNet

from tests.tamakEriFever.config import USE_BZ2


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        traj_queue = mem.TrajQueueMP(maxlen=32,
                                     queue_sender=queue_sender,
                                     batch_size=16,
                                     use_bz2=USE_BZ2,
                                     num_batch_maker=8,
                                     logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"))

        memory_server = mem.MemoryServer(traj_queue, self.port, actor_num=None,
                                         tensorboard_dir=os.path.join(self.logger_file_dir, "sample_reward"))
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, qqueue_senders):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor_receiver = self.create_receiver(queue_receiver, device=device)
        model = FootballNet().to(device)
        impala = alg.IMPALA(model, tensor_receiver,
                            lr=0.00019896, gamma=0.993, lbd=1, vf=0.5, ef=0.00087453,
                            queue_senders=qqueue_senders,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            vtrace_key=["checkpoints"])
        impala.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        league = lg.League(queue_receiver, self.port,
                           cache_weights_intervals=1,
                           save_weights_dir=os.path.join(self.logger_file_dir, "model"),
                           save_weights_intervals=1000,
                           tensorboard_dir=os.path.join(self.logger_file_dir, "eval_info"),
                           use_bz2=USE_BZ2)
        league.run()
