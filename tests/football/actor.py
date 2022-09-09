
from tests.football.football_env import OpponentEnv
from tests.football.models import feature_model, tamak_model
from rl.agent import IMPALAAgent
import rl.core as core


class ActorMain(core.ActorMainBase):
    def create_env_and_agents_pool(self, gather_id: int, actor_id: int):
        agents_pool = {
            "builtin_ai": feature_model.BuiltinAI(name="builtin_ai"),
            "tamak": tamak_model.TamakAgent(model=tamak_model.FootballNet(name="tamak")),
            "feature": IMPALAAgent(model=feature_model.FeatureModel("feature"))
        }

        env = OpponentEnv(agents_pool={
            "builtin_ai": feature_model.BuiltinAI(name="builtin_ai"),
            "tamak": tamak_model.TamakAgent(model=tamak_model.FootballNet(name="tamak")),
            "feature": IMPALAAgent(model=feature_model.FeatureModel("feature"))
        })
        return env, agents_pool





