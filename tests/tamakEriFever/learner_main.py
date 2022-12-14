from tests.tamakEriFever.learner import MemoryMain, LearnerMain, LeagueMain
from rl.core import train_main
import tests.tamakEriFever.config as cfg

if __name__ == '__main__':

    mr_main1 = MemoryMain(port=cfg.MEMORY1_ADDRESS[1], logger_file_dir=f"./log/{cfg.NAME}/")
    league_main = LeagueMain(port=cfg.LEAGUE_ADDRESS[1], logger_file_dir=f"./log/{cfg.NAME}/")
    leaner_main = LearnerMain(logger_file_dir=f"./log/{cfg.NAME}/")

    train_main(leaner_main, [mr_main1], [league_main], memory_buffer_length=8)
