from tests.impala.actor import ActorCreate
from rl.core import open_gather

if __name__ == "__main__":
    logger_file_dir = "./log/lunar_lander/gathers"
    actor_main = ActorCreate(num_steps=32, logger_file_dir=logger_file_dir)

    open_gather(2, ("192.168.43.157", 7777),
                ("192.168.43.157", 7779),
                num_actors=2, actor_roles="sampler", func=actor_main,
                use_bz2=False, logger_file_dir=logger_file_dir)

