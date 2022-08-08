from tests.football.actor import ActorCreate
from rl.core import open_gather

import tests.football.config as cfg


if __name__ == "__main__":
    logger_file_dir = f"./log/{cfg.NAME}/gathers/"
    actor_main = ActorCreate(num_steps=32, logger_file_dir=logger_file_dir)

    open_gather(1, memory_server_address=cfg.MEMORY_ADDRESS, league_address=cfg.LEAGUE_ADDRESS,
                num_actors=2, actor_roles="sampler", func=actor_main, use_bz2=True, logger_file_dir=logger_file_dir)
