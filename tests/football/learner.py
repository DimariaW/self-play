import pickle
import torch
import multiprocessing as mp
import os
import rl.core as core
import rl.memory as mem
import rl.algorithm as alg
import rl.league as lg
import logging

from tests.football.football_model import CNNModel

from tests.football.config import USE_BZ2


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        logging.info(f"the data will be in {device}")

        traj_queue = mem.TrajQueueMP(maxlen=32,
                                     queue_sender=queue_sender,
                                     batch_size=32,
                                     use_bz2=USE_BZ2,
                                     to_tensor=True,
                                     device=device,
                                     num_batch_maker=8,
                                     logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                     logger_file_level=logging.DEBUG)

        memory_server = mem.MemoryServer(traj_queue, self.port, actor_num=None)
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        logging.info(f"the data will be in {device}")

        tensor_receiver = self.create_receiver(queue_receiver, to_tensor=False)

        model = CNNModel((16, 72, 96), 19, name="cnn").to(device)
        model_weights = pickle.load(open("./tests/football/models/smm_cnn.pickle", "rb"))
        if type(model_weights) == bytes:
            model_weights = pickle.loads(model_weights)
        model.set_weights(model_weights)
        logging.info("successfully loads weight from pretrained !")

        impala = alg.IMPALA(model,
                            queue_senders,
                            tensor_receiver,
                            lr=0.00019896, gamma=0.993, lbd=1, vf=0.5, ef=0.00087453,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "train_info"),
                            vtrace_key=["checkpoints"])
        impala.run()


class ModelServerMain(core.ModelServerMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        model_server = lg.ModelServer(queue_receiver, self.port, use_bz2=USE_BZ2, cache_weights_intervals=1)
        model_server.run()


class LeagueMain(core.ModelServerMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        league = lg.ModelServer4RecordAndEval(queue_receiver, self.port, use_bz2=USE_BZ2,
                                              cache_weights_intervals=10000,
                                              save_weights_dir=os.path.join(self.logger_file_dir, "models"),
                                              tensorboard_dir=os.path.join(self.logger_file_dir, "metrics"))
        league.run()


