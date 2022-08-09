import logging
import queue
import time
import torch
import numpy as np
import multiprocessing as mp

import rl.core as core
import rl.memory as mm


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_receiver: mp.Queue):

        traj_queue = mm.TrajQueueMP(maxlen=16,
                                    queue_sender=queue_receiver,
                                    batch_size=16,
                                    num_batch_maker=3,
                                    use_bz2=False,
                                    to_tensor=True,
                                    logger_file_dir="./log/test_traj_queue_mp/")
        traj_queue.start()
        for j in range(64 * 16):
            episode = [
                    {f"key{i}": np.random.randn(11, 11) for i in range(20)}
                    for _ in range(64)
                        ]
            traj_queue.cache([(j, episode)])
        traj_queue.stop()
        time.sleep(10)


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        tensor_receiver = self.create_receiver(queue_receiver, num_sender=3)
        num = 0
        try:
            while True:
                _, tensor = tensor_receiver.recv()
                num += 1
                logging.debug(f"successfully receive {num}")
                time.sleep(0.01)
                queue_senders[0].put((False, tensor))
                logging.debug(f"successfully put {num}")
        except queue.Empty:
            logging.debug("end receiving")
            queue_senders[0].put((True, None))


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_sender: mp.Queue):
        receiver = self.create_receiver(queue_sender, num_sender=1)
        num = 0
        try:
            while True:
                data = receiver.recv()
                num += 1
                logging.info(f"receive num {num}")
                logging.info(list(data.keys()))
        except queue.Empty:
            logging.info("end receiving")
