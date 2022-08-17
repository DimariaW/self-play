import multiprocessing as mp
from tests.test_tensorboard.test_utils import main
if __name__ == "__main__":
   mp.Process(target=main, args=()).start()
