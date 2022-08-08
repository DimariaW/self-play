from tests.test_league.test_league import LeagueMain, LearnerMain, MemoryMain

from rl.core import train_main

if __name__ == "__main__":
    mm = MemoryMain(port=-1)
    league_main = LeagueMain(port=8889, logger_file_dir="./log/")
    learner_main = LearnerMain(logger_file_dir="./log/")

    train_main(learner_main, [mm], [league_main], memory_buffer_length=1)