import rl.utils as utils



@utils.wrap_traceback
def main():
    utils.set_process_logger()
    raise RuntimeError("test error")

import pickle

pickle.load()