import pickle
import bz2
import torch
import multiprocessing as mp
import os
import rl.core as core
import rl.memory as mem
import rl.algorithm as alg
import rl.league as lg
import numpy as np
import logging

from tests.football.football_model import CNNModel

from tests.football.config import USE_BZ2


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"the data will be in {device}")

        traj_queue = mem.TrajQueueMP(maxlen=16,
                                     queue_sender=queue_sender,
                                     batch_size=16,
                                     use_bz2=USE_BZ2,
                                     to_tensor=True,
                                     device=device,
                                     num_batch_maker=1,
                                     logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"))

        memory_server = mem.MemoryServer(traj_queue, self.port, actor_num=None,
                                         tensorboard_dir=None)
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        logging.info(f"the data will be in {device}")

        tensor_receiver = self.create_receiver(queue_receiver, to_tensor=False)

        model = CNNModel((16, 72, 96), 19, name="cnn").to(device)
        # compressed_model_weights = np.load("./tests/football/kaggle_model/model_785914.npy", allow_pickle=True).item()
        # model.set_weights(pickle.loads(bz2.decompress(compressed_model_weights)))
        # logging.info("successfully loads weight from pretrained !")

        impala = alg.IMPALA(model, tensor_receiver,
                            lr=0.00019896, gamma=0.993, lbd=1, vf=0.5, ef=0.00087453,
                            queue_senders=queue_senders,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            vtrace_key=["checkpoints"])
        impala.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        league = lg.League(queue_receiver, self.port,
                           cache_weights_intervals=1,
                           save_weights_dir=os.path.join(self.logger_file_dir, "model"),
                           save_weights_intervals=10000,
                           tensorboard_dir=os.path.join(self.logger_file_dir, "eval_info"),
                           use_bz2=USE_BZ2)
        league.add_opponent("builtin_ai", index=0, weights={}, winning_rate=0.)
        league.run()
