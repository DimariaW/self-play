from tests.PPO.async_impala.config import CONFIG, MEMORY_ADDRESS, MODEL_SERVER_ADDRESS, LEAGUE_ADDRESS
from tests.PPO.async_impala.learner import MemoryMain, LearnerMain, LeagueMain, ModelServerMain
from rl.core import train_main
import logging

if __name__ == '__main__':

    mm_main = MemoryMain(MEMORY_ADDRESS[1], f"./log/{CONFIG['name']}/", logger_file_level=logging.INFO)

    model_server_main = ModelServerMain(MODEL_SERVER_ADDRESS[1],
                                        logger_file_dir=f"./log/{CONFIG['name']}/", logger_file_level=logging.INFO)

    league_main = LeagueMain(LEAGUE_ADDRESS[1],
                             logger_file_dir=f"./log/{CONFIG['name']}/", logger_file_level=logging.INFO)

    leaner_main = LearnerMain(f"./log/{CONFIG['name']}/", logger_file_level=logging.INFO)

    train_main(leaner_main, [mm_main], [model_server_main, league_main], memory_buffer_length=8)
