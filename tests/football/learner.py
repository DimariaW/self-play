import torch
import multiprocessing as mp
import os
import pickle

import rl.core as core
import rl.memory as mem
import rl.algorithm as alg
import rl.league as lg


from tests.football.config import CONFIG


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        device = torch.device("cpu")

        if CONFIG["memory_type"] == "list":
            traj_list = mem.TrajListMP(maxlen=CONFIG["maxlen"], queue_sender=queue_sender,
                                       batch_size=CONFIG["batch_size"], priority_replay=CONFIG["priority_replay"],
                                       to_tensor=True, device=device,
                                       # batch_maker args
                                       num_batch_maker=36,
                                       logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                       logger_file_level=self.logger_file_level)
        else:
            traj_list = mem.TrajQueueMP(maxlen=64, queue_sender=queue_sender,
                                        batch_size=64, use_bz2=CONFIG["use_bz2"],
                                        to_tensor=True, device=device,
                                        # batch_maker args
                                        num_batch_maker=8,
                                        logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                        logger_file_level=self.logger_file_level)

        memory_server = mem.MemoryServer(traj_list, self.port, actor_num=None)
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor_receiver = self.create_receiver(queue_receiver, to_tensor=True, device=device)

        env_model_config = CONFIG["env_model_config"]
        model = env_model_config["model"]().to(device)

        impala = alg.PPO(model, queue_senders, tensor_receiver,
                         lr=1e-4, gamma=0.993, lbd=0.95, vf=1, ef=1e-3,
                         tensorboard_dir=os.path.join(self.logger_file_dir, "learn_info"),
                         max_update_num_per_seconds=CONFIG["max_update_num_per_seconds"],
                         critic_key=["checkpoints"],
                         critic_update_method=CONFIG["critic_update_method"],
                         using_critic_update_method_adv=CONFIG["using_critic_update_method_adv"],
                         actor_key=["checkpoints"],
                         actor_update_method=CONFIG["actor_update_method"],
                         )

        weights = pickle.load(open("./env_models/football/weights/feature_vs_tamak_70.pickle", "rb"))
        impala.set_weights(weights, 0)
        impala.run()


class ModelServerMain(core.ModelMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)
        model_server = lg.ModelServer(queue_receiver, self.port, use_bz2=CONFIG["use_bz2"], cache_weights_intervals=1)
        model_server.run()


class LeagueMain(core.LeagueMainBase):
    def main(self, queue_receiver: mp.Queue):
        queue_receiver = self.create_receiver(queue_receiver)

        league = lg.ModelServer4Evaluation(queue_receiver, self.port, use_bz2=CONFIG["use_bz2"],
                                           cache_weights_intervals=1000,
                                           save_weights_dir=os.path.join(self.logger_file_dir, "model_weights"),
                                           tensorboard_dir=os.path.join(CONFIG["metrics_dir"], CONFIG["name"]))
        league.run()

