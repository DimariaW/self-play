import torch
import gym
import rl.core as core


class ActorMain(core.ActorMainBase):
    def create_env_and_agent(self, gather_id: int, actor_id: int):
        from tests.test_env_models.env_model_gym import EnvWrapper, Model, ModelLSTM
        from tests.impala.config import CONFIG
        from rl.agent import IMPALAAgent

        env = gym.make(CONFIG["env_name"])
        env = EnvWrapper(env, reward_threshold=CONFIG["reward_threshold"])
        device = torch.device("cpu")
        model = Model(CONFIG["obs_dim"], CONFIG["num_act"], use_orthogonal_init=True, use_tanh=False).to(device)
        # model = ModelLSTM(CONFIG["obs_dim"], CONFIG["num_act"]).to(device)
        agent = IMPALAAgent(model, device)
        return env, agent

