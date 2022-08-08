import rl.core as core
import rl.league as lg
import rl.model as model_base

import os
import torch
import torch.nn as nn
import time
import logging
import multiprocessing as mp
from typing import List, Union, Tuple


class Model(model_base.Model):
    def __init__(self, name="test"):
        super(Model, self).__init__(name)
        self.fc = nn.Linear(2, 2)

    def forward(self, state):
        return self.fc(state)


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders: Union[List[mp.Queue], Tuple[mp.Queue]]):
        model = Model()
        num_update = torch.tensor(0)
        queue_senders[0].put((False, (num_update, model)))

        while True:
            time.sleep(5)
            torch.nn.init.kaiming_normal_(model.fc.weight)
            logging.debug(f"model.fc.weights is {model.fc.weight}")
            num_update += 1


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_receiver: mp.Queue):
        receiver = self.create_receiver(queue_receiver, num_sender=-1)
        league = lg.League(receiver, port=self.port, use_bz2=False, cache_weights_intervals=1, save_weights_intervals=3,
                           save_weights_dir=os.path.join(self.logger_file_dir, "model"),
                           tensorboard_dir=os.path.join(self.logger_file_dir, "metric"))
        league.run()


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        pass

