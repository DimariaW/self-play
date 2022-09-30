# env
the env protocol should follow gym new api style
- reset(*args, **kwargs) -> obs
  - reset 支持参数传入，如在self-play模式下，env中需要传入对手信息
- step(action) -> next_obs, reward_info, done, truncated, info
  - reward_info: dict[str, float], support multi reward head
  - done: episode end
  - truncated: reach max time length
  - info: other useful information
- when done or truncated, the actor will send infos to league。
  infos will automatically contain the cumulative episodes reward(no discount factor)
  episode steps and meta_info that contains info.
  - when necessary, you can define a function to extract infos from meta_info 
    that will be loaded by tensorboard.

# model
currently support ModelValueLogit that get batched tensor and return value_info and logits