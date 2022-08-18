from tests.impala.actor import ActorMain
from tests.impala.config import CONFIG, LEAGUE_ADDRESS, MEMORY_ADDRESS, MODEL_SERVER_ADDRESS, USE_BZ2
from rl.core import open_gather

if __name__ == "__main__":
    logger_file_dir = f"./log/{CONFIG['env_name'] + '_lstm'}/actors/"
    actor_main = ActorMain(num_steps=32, logger_file_dir=logger_file_dir)

    open_gather(num_gathers=8,
                roles=["sampler"]*8,
                num_actors=2,
                league_address=LEAGUE_ADDRESS,
                memory_server_address=MEMORY_ADDRESS,
                model_server_address=MODEL_SERVER_ADDRESS,
                actor_main=actor_main,
                use_bz2=USE_BZ2,
                self_play=False,
                logger_file_dir=logger_file_dir)
