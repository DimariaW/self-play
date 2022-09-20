import logging
import queue
import time
import os
import torch
import numpy as np
import multiprocessing as mp

import rl.core as core
import rl.memory as mm


class MemoryMain(core.MemoryMainBase):
    def main_old(self, queue_receiver: mp.Queue):

        traj_queue = mm.TrajQueueMP(maxlen=16,
                                    queue_sender=queue_receiver,
                                    batch_size=16,
                                    num_batch_maker=3,
                                    use_bz2=False,
                                    to_tensor=True,
                                    logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                    logger_file_level=self.logger_file_level)
        traj_queue.start()
        for j in range(64 * 16):
            episode = [
                    {f"key{i}": np.random.randn(11, 11) for i in range(20)}
                    for _ in range(64)
                        ]
            traj_queue.cache([(j, episode)])

        traj_queue.stop()
        time.sleep(10)

    def main(self, queue_sender: mp.Queue):
        import bz2
        import pickle
        traj_list = mm.TrajListMP(maxlen=300, queue_sender=queue_sender, batch_size=64, priority_replay=False,
                                  to_tensor=True, num_batch_maker=3,
                                  logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                  logger_file_level=self.logger_file_level)
        traj_list.start()

        for j in range(500):
            episodes = [
                    {f"key{i}": np.random.randn(11, 11) for i in range(20)}
                    for _ in range(np.random.randint(50, 80))
                        ]
            episodes = [bz2.compress(pickle.dumps(episodes[i:i + 16]))
                        for i in range(0, len(episodes), 16)]
            traj_list.cache([(j, episodes)])

        while True:
            time.sleep(10)


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        tensor_receiver = self.create_receiver(queue_receiver, num_sender=3)
        num = 0
        try:
            while True:
                index, tensor = tensor_receiver.recv()
                num += 1
                logging.info(f"{index}")
                logging.info(f"{list(tensor.keys())}, {tensor[list(tensor.keys())[0]].shape}")
                logging.info(f"successfully receive {num}")
        except queue.Empty:
            logging.info("end receiving")


class LeagueMain(core.ModelServerMainBase):
    def main(self, queue_receiver: mp.Queue):
        pass
