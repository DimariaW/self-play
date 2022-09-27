import rl.actor as actor
from tests.PPO.config import CONFIG
from rl.agent import PPOAgent


class ActorMain(actor.ActorMainBase):
    def main(self, gather_id, actor_id, role, queue_gather2actor, queue_actor2gather):
        if role == "sampler":
            actor_sampler = self.create_sampler_actor()
            actor_manager = actor.ActorSamplerManager(actor_id, actor_sampler, queue_gather2actor, queue_actor2gather,
                                                      use_bz2=CONFIG["use_bz2"],
                                                      compress_step_length=CONFIG["compressed_step_length"],
                                                      self_play=CONFIG["self-play"])
            actor_manager.run()
        elif role == "evaluator":
            actor_evaluator = self.create_evaluator_actor()
            actor_manager = actor.ActorEvaluatorManager(actor_id, actor_evaluator, queue_gather2actor, queue_actor2gather,
                                                        use_bz2=CONFIG["use_bz2"], self_play=CONFIG["self-play"])
            actor_manager.run()

    @staticmethod
    def create_sampler_actor():
        env_model_config = CONFIG["env_model_config"]
        env = env_model_config["env_class"](**env_model_config["env_args"])
        model = env_model_config["model_class"](**env_model_config["model_args"])

        agent = PPOAgent(model)
        agents_pool = {agent.model_id[0]: agent}
        actor_sampler = actor.ActorSampler(env, agents_pool,
                                           num_steps=CONFIG["num_steps"], get_full_episodes=CONFIG["get_full_episodes"],
                                           postprocess_traj=actor.PostProcess.get_cal_gae_func(gamma_infos=0.99,
                                                                                               lbd_infos=0.98))
        return actor_sampler

    @staticmethod
    def create_evaluator_actor():
        env_model_config = CONFIG["env_model_config"]
        env = env_model_config["env_class"](**env_model_config["env_args"])
        model = env_model_config["model_class"](**env_model_config["model_args"])

        agent = PPOAgent(model)
        agents_pool = {agent.model_id[0]: agent}
        actor_evaluator = actor.ActorEvaluator(env, agents_pool, num_episodes=1)

        return actor_evaluator
