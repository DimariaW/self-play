import gym
import collections
from typing import Tuple, Dict, Any, Optional, Callable
import gfootball.env as gfootball_env
import random
from rl.agent import Agent
import rl.utils as utils
import numpy as np


class Env(gym.Wrapper):
    """
    multi-agent env that all agent are homogeneous,
    so we simply treat different agent as batch.

    in the env, an agents_pool and agents_to_process_func is maintained,
    an item of agents_pool is model_name:str -> agent: Agent

    when reset, current model_id and (opponent_id with weights) is sent.
    model_id only used the model_name to select preprocess func and model_index is ignored.
    """
    def __init__(self, env_name, num_left, num_right):
        env = gfootball_env.create_environment(env_name=env_name,
                                               render=False,
                                               representation="raw",
                                               rewards="scoring,checkpoints",
                                               number_of_left_players_agent_controls=num_left,
                                               number_of_right_players_agent_controls=num_right,
                                               other_config_options={"action_set": "v2"})
        super().__init__(env)
        self.agents_pool = {}
        self.agents_to_process_func = {}
        # opponent info
        self.opponent_agent = None
        self.opponent_preprocess_func = None
        self.opponent_action_histories = [collections.deque(maxlen=8) for _ in range(num_right)]
        self.opponent_obs = None
        # current info
        self.preprocess_func = None
        self.action_histories = [collections.deque(maxlen=8) for _ in range(num_left)]
        self.scoring = 0
        self.num_left = num_left
        self.num_right = num_right

    def add_agent(self, name: str, agent: Agent, process_func: Callable[[dict, collections.deque], dict] = None):
        assert name not in self.agents_pool
        assert name == agent.model_id[0]
        self.agents_pool[name] = agent
        self.agents_to_process_func[name] = process_func

    def _reset(self):
        obs = self.env.reset()
        for _ in range(random.randint(0, 50)):
            obs, reward, done, info = self.env.step([0] * (self.num_right + self.num_left))
            if done:
                obs = self.env.reset()
        # preprocess
        for action_history in self.action_histories:
            action_history.extend([0] * 8)
        feature = [self.preprocess_func(observation, action_history)
                   for observation, action_history in zip(obs[:self.num_left], self.action_histories)]

        for action_history in self.opponent_action_histories:
            action_history.extend([0] * 8)
        opponent_feature = [self.opponent_preprocess_func(observation, action_history)
                            for observation, action_history in zip(obs[self.num_left:], self.opponent_action_histories)]
        self.opponent_obs = utils.batchify(opponent_feature, unsqueeze=0)
        self.scoring = 0
        return utils.batchify(feature, unsqueeze=0)

    def reset(self, model_id: Tuple[str, Optional[int]],
              opponent_id: Tuple[str, Optional[int]],
              opponent_weights: Optional):
        # set agent
        model_name, _ = model_id
        self.preprocess_func = self.agents_to_process_func[model_name]
        # set opponent agent
        opponent_name, opponent_index = opponent_id
        self.opponent_agent = self.agents_pool[opponent_name]
        self.opponent_agent.set_weights(opponent_weights, opponent_index)
        self.opponent_preprocess_func = self.agents_to_process_func[opponent_name]
        # reset
        return self._reset()

    def step(self, action) -> Tuple[Any, Dict[str, float], bool, bool, Dict]:
        # load action
        for act, action_history in zip(action, self.action_histories):
            action_history.append(act)

        opponent_action = self.opponent_agent.predict(utils.batchify([self.opponent_obs], unsqueeze=0))["action"][0]
        for act, action_history in zip(opponent_action, self.opponent_action_histories):
            action_history.append(act)
        # step
        obs, reward, done, info = self.env.step(np.concatenate([action, opponent_action]))
        obs: list
        reward: np.ndarray
        done: bool
        info: dict
        reward_infos = {"checkpoints": reward[:self.num_left],
                        "scoring": np.array([info["score_reward"]] * self.num_left)}
        # preprocess
        feature = [self.preprocess_func(observation, action_history)
                   for observation, action_history in zip(obs[:self.num_left], self.action_histories)]
        opponent_feature = [self.opponent_preprocess_func(observation, action_history)
                            for observation, action_history in zip(obs[self.num_left:], self.opponent_action_histories)]
        self.opponent_obs = utils.batchify(opponent_feature, unsqueeze=0)
        # record
        self.scoring += info["score_reward"]
        truncated = False
        if done:
            if self.scoring == 0:
                info["win"] = 0.5
            elif self.scoring > 0:
                info["win"] = 1
            else:
                info["win"] = 0
            info["opponent_id"] = self.opponent_agent.model_id
        return utils.batchify(feature, unsqueeze=0), reward_infos, done, truncated, info


class FixOpponentEnv(Env):
    def set_opponent(self, model_id: Tuple[str, Optional[int]],
                     opponent_id: Tuple[str, Optional[int]],
                     opponent_weights: Optional):
        # set agent
        model_name, _ = model_id
        self.preprocess_func = self.agents_to_process_func[model_name]
        # set opponent agent
        opponent_name, opponent_index = opponent_id
        self.opponent_agent = self.agents_pool[opponent_name]
        self.opponent_agent.set_weights(opponent_weights, opponent_index)
        self.opponent_preprocess_func = self.agents_to_process_func[opponent_name]

    def reset(self,
              model_id: Tuple[str, Optional[int]] = None,
              opponent_id: Tuple[str, Optional[int]] = None,
              opponent_weights: Optional = None):
        return self._reset()


