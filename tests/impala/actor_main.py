from tests.impala.actor import ActorMain
from tests.impala.config import CONFIG, LEAGUE_ADDRESS, MEMORY_ADDRESS, MODEL_SERVER_ADDRESS
from rl.core import open_gather
import logging

if __name__ == "__main__":

    logger_file_dir = f"./log/{CONFIG['name']}/actors/"
    actor_main = ActorMain(logger_file_dir=logger_file_dir, logger_file_level=logging.INFO)

    open_gather(num_gathers=8,
                roles=["sampler"]*7 + ["evaluator"],
                num_actors=2,
                league_address=LEAGUE_ADDRESS,
                memory_server_address=MEMORY_ADDRESS,
                model_server_address=MODEL_SERVER_ADDRESS,
                actor_main=actor_main,
                logger_file_dir=logger_file_dir,
                logger_file_level=logging.INFO)
