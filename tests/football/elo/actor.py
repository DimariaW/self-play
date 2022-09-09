from tests.football.actor import ActorMain
from rl.core import open_gather


if __name__ == "__main__":
    logger_file_dir = f"./log/elo_infos/gathers/"
    actor_main = ActorMain(logger_file_dir=logger_file_dir)

    open_gather(num_gathers=20,
                roles='evaluator',
                num_actors=2,
                league_address=("10.127.7.14", 56789),
                # league_address=("172.18.237.19", 56789),
                actor_main=actor_main,
                use_bz2=True,
                self_play=True,
                logger_file_dir=logger_file_dir)
