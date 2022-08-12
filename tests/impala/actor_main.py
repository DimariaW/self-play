from tests.impala.actor import ActorCreate
from tests.impala.config import CONFIG, LEAGUE_ADDRESS, MEMORY_ADDRESS, MODEL_SERVER_ADDRESS, USE_BZ2
from rl.core import open_gather

if __name__ == "__main__":
    logger_file_dir = f"./log/{CONFIG['env_name']}/actors/"
    actor_main = ActorCreate(num_steps=32, logger_file_dir=logger_file_dir)

    open_gather(num_gathers=2,
                league_address=LEAGUE_ADDRESS,
                memory_server_address=MEMORY_ADDRESS,
                model_server_address=MODEL_SERVER_ADDRESS,
                num_actors=2,
                actor_roles="sampler",
                func=actor_main,
                use_bz2=USE_BZ2,
                self_play=False,
                logger_file_dir=logger_file_dir)
    """
    open_gather(num_gathers=1,
                league_address=LEAGUE_ADDRESS,
                memory_server_address=None,
                model_server_address=None,
                num_actors=1,
                actor_roles="evaluator",
                func=actor_main,
                use_bz2=USE_BZ2,
                self_play=False,
                logger_file_dir=logger_file_dir)
    """