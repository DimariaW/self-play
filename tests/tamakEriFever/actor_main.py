from tests.tamakEriFever.actor import ActorCreate
from rl.core import open_gather

import tests.tamakEriFever.config as cfg


if __name__ == "__main__":
    memory_server_index = 1  # 2, -1
    logger_file_dir = f"./log/{cfg.NAME}/gathers_{memory_server_index}/"
    actor_main = ActorCreate(num_steps=32, logger_file_dir=logger_file_dir)
    open_gather(11, memory_server_address=cfg.MEMORY1_ADDRESS, league_address=cfg.LEAGUE_ADDRESS,
                num_actors=4, actor_roles="sampler", func=actor_main, use_bz2=True, logger_file_dir=logger_file_dir)
