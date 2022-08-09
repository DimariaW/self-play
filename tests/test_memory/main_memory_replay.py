from rl.core import train_main

from tests.test_memory.test_memory_replay import MemoryMain, LeagueMain, LearnerMain

if __name__ == '__main__':
    name = ["test_traj_list", "test_traj_queue", "test_traj_queue_mp"][2]
    mr_main = MemoryMain(port=-1, logger_file_dir=f"./log/{name}/")
    league_main = LeagueMain(port=-1, logger_file_dir=f"./log/{name}/")
    leaner_main = LearnerMain(logger_file_dir=f"./log/{name}/")

    train_main(leaner_main, [mr_main], [league_main])
