import torch
import multiprocessing as mp
import os

import rl.core as core
import rl.memory as mem
import rl.algorithm as alg
import rl.league as lg


from tests.PPO.async_impala.config import CONFIG


class MemoryMain(core.MemoryMainBase):
    def main(self, queue_sender: mp.Queue):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if CONFIG["get_full_episodes"]:
            traj_list = mem.TrajListMP(maxlen=3000, queue_sender=queue_sender,
                                       batch_size=64, priority_replay=False,
                                       to_tensor=True, device=device,
                                       # batch_maker args
                                       num_batch_maker=2,
                                       logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                       logger_file_level=self.logger_file_level)
        else:
            traj_list = mem.TrajQueueMP(maxlen=8, queue_sender=queue_sender,
                                        batch_size=16, use_bz2=CONFIG["use_bz2"],
                                        to_tensor=True, device=device,
                                        # batch_maker args
                                        num_batch_maker=2,
                                        logger_file_dir=os.path.join(self.logger_file_dir, "batch_maker"),
                                        logger_file_level=self.logger_file_level)

        memory_server = mem.MemoryServer(traj_list, self.port, actor_num=None)
        memory_server.run()


class LearnerMain(core.LearnerMainBase):
    def main(self, queue_receiver: mp.Queue, queue_senders):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor_receiver = self.create_receiver(queue_receiver)

        env_model_config = CONFIG["env_model_config"]
        model = env_model_config["model_class"](**env_model_config["model_args"]).to(device)

        impala = alg.IMPALA(model, queue_senders, tensor_receiver,
                            lr=1e-4, gamma=0.99, lbd=0.98, vf=0.5, ef=1e-4,
                            tensorboard_dir=os.path.join(self.logger_file_dir, "learn_info"),
                            critic_key=["reward"],
                            vtrace_key=["reward"],
                            critic_update_method=CONFIG["critic_update_method"],
                            upgo_key=CONFIG["upgo_key"],
                            )
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
