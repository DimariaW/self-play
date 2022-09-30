
from env_models import cartpole
import numpy as np

env_model_factory = cartpole.ENV_MODELS["MountainCarContinuous-FC"]

env = env_model_factory["env"]()

model = env_model_factory["model"]()

obs = env.reset()

obs, reward, done, truncated, info = env.step(np.array([5]))

