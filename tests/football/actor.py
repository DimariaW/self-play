import tests.football.football_env as football_env
from tests.football.models import feature_model, tamak_model
import tests.football.config as cfg

from rl.agent import IMPALAAgent
import rl.core as core
import rl.actor as actor
import pickle


class ActorMain(core.ActorMainBase):
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
            actor_manager = actor.ActorEvaluatorManager(actor_id, actor_evaluator, queue_gather2actor,
                                                        queue_actor2gather,
                                                        use_bz2=CONFIG["use_bz2"], self_play=CONFIG["self-play"])
            actor_manager.run()

    @staticmethod
    def create_opponent_env_and_agents_pool():
        agents_pool = {
            "builtin_ai": feature_model.BuiltinAI(name="builtin_ai"),
            "tamak": tamak_model.TamakAgent(model=tamak_model.FootballNet(name="tamak")),
            "feature": IMPALAAgent(model=feature_model.FeatureModel("feature"))
        }

        env = football_env.OpponentEnv(agents_pool={
            "builtin_ai": feature_model.BuiltinAI(name="builtin_ai"),
            "tamak": tamak_model.TamakAgent(model=tamak_model.FootballNet(name="tamak")),
            "feature": IMPALAAgent(model=feature_model.FeatureModel("feature"))
        })
        return env, agents_pool

    @staticmethod
    def create_tamak_opponent_env_and_agents_pool():
        agents_pool = {
            "feature": IMPALAAgent(model=feature_model.FeatureModel("feature"))
        }

        opponent_agent = tamak_model.TamakAgent(model=tamak_model.FootballNet(name="tamak"))
        opponent_agent.set_weights(weights=pickle.load(open("./tests/football/weights/tamak_1679.pickle", "rb")),  # ./tests/football/
                                   index=1679)

        env = football_env.SingleOpponentEnv(
            opponent_agent=opponent_agent,
            opponent_preprocess_func=football_env.TamakEriFever.preprocess_obs,
            preprocess_func=football_env.Observation2Feature.preprocess_obs
        )
        return env, agents_pool




