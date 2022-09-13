import tests.football.football_env as football_env
from tests.football.models import feature_model, tamak_model
import tests.football.config as cfg

from rl.agent import IMPALAAgent
import rl.core as core

import pickle


class ActorMain(core.ActorMainBase):
    def create_env_and_agents_pool(self, gather_id: int = None, actor_id: int = None):
        if cfg.SELF_PLAY:
            return self._create_self_play_env_and_agents_pool()
        else:
            return self._create_single_opponent_env_and_agents_pool()

    @staticmethod
    def _create_self_play_env_and_agents_pool():
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
    def _create_single_opponent_env_and_agents_pool():
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



