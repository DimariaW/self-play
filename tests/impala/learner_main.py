from tests.impala.config import CONFIG, MEMORY_ADDRESS, MODEL_SERVER_ADDRESS, LEAGUE_ADDRESS
from tests.impala.learner import MemoryMain, LearnerMain, LeagueMain, ModelServerMain
from rl.core import train_main


if __name__ == '__main__':
    name = CONFIG["env_name"] + "_lstm"

    mm_main = MemoryMain(MEMORY_ADDRESS[1], f"./log/{name}/")

    model_server_main = ModelServerMain(MODEL_SERVER_ADDRESS[1], name="model_server", logger_file_dir=f"./log/{name}/")
    league_main = LeagueMain(LEAGUE_ADDRESS[1], name="league", logger_file_dir=f"./log/{name}/")

    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, [mm_main], [model_server_main, league_main], memory_buffer_length=8)
