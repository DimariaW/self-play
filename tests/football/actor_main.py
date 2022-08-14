from tests.football.actor import ActorMain
from rl.core import open_gather

import tests.football.config as cfg


if __name__ == "__main__":
    logger_file_dir = f"./log/{cfg.NAME}/gathers/"
    actor_main = ActorMain(num_steps=32, logger_file_dir=logger_file_dir)

    open_gather(num_gathers=17, roles='sampler', num_actors=3,
                league_address=cfg.LEAGUE_ADDRESS,
                memory_server_address=cfg.MEMORY_ADDRESS,
                model_server_address=cfg.MODEL_ADDRESS,
                actor_main=actor_main,
                use_bz2=cfg.USE_BZ2,
                self_play=False,
                logger_file_dir=logger_file_dir)
