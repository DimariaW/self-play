from tests.football.learner import MemoryMain, LearnerMain, LeagueMain, ModelServerMain
from rl.core import train_main
import tests.football.config as cfg

if __name__ == '__main__':
    mr_main = MemoryMain(port=cfg.MEMORY_ADDRESS[1], logger_file_dir=f"./log/{cfg.NAME}/")

    model_server_main = ModelServerMain(port=cfg.MODEL_ADDRESS[1], name="model_server",
                                        logger_file_dir=f"./log/{cfg.NAME}/")
    league_main = LeagueMain(port=cfg.LEAGUE_ADDRESS[1], name="league",
                             logger_file_dir=f"./log/{cfg.NAME}/")

    leaner_main = LearnerMain(logger_file_dir=f"./log/{cfg.NAME}/")

    train_main(leaner_main, [mr_main], [model_server_main, league_main], memory_buffer_length=8)



