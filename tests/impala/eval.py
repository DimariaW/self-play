from tests.impala.actor import ActorMain
import rl.actor as rl_actor
import rl.utils as utils
import logging
import pickle
import gym


class RenderWrapper(gym.Wrapper):
    def reset(self):
        obs = self.env.reset()
        self.env.render()
        return obs

    def step(self, action):
        info = self.env.step(action)
        self.env.render()
        return info


utils.set_process_logger()
env, agent = ActorMain().create_env_and_agent(0, 0)
env = RenderWrapper(env)

actor = rl_actor.Actor(env, agent, num_episodes=100)

weights = pickle.load(open("./models/cnn_1500000.pickle", "rb"))
index = 400000

actor.reset_agent(("nodel", index), weights)
# actor.reset_env(("builtin_ai", None), None)
actor.reset_env()

for _, data in actor.predict():
    logging.info(data)
    # actor.reset_env(("builtin_ai", None), None)
    actor.reset_env()
