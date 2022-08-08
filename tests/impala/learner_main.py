
from tests.impala.learner import MemoryMain, LearnerMain, LeagueMain
from rl.core import train_main


if __name__ == '__main__':
    name = "lunar_lander"
    mm_main1 = MemoryMain(7777, f"./log/{name}/mm1")
    league_main = LeagueMain(7779, f"./log/{name}/")
    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, [mm_main1], [league_main], memory_buffer_length=4)
