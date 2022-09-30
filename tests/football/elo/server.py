import pickle
import rl.utils as utils
from rl.league import ELOEvaluator
import os
from typing import Dict, Any


def load_models_from_dir(dir_name):
    files = os.listdir(dir_name)
    files = [file for file in files if file.endswith(".pickle")]
    models_pool: Dict[Any, Any] = {}
    for file in files:
        model_name, model_index = file.rstrip(".pickle").split("_")
        weights = pickle.load(open(os.path.join(dir_name, file), "rb"))
        models_pool[model_name, int(model_index)] = weights
    return models_pool


def main():
    log_dir = "./log/elo_infos/server.txt"
    utils.set_process_logger(file_path=log_dir)

    dir_name = "./log/self_play_feature_checkpoints_impala/deprecated"
    # dir_name = "../weights"
    models_pool = load_models_from_dir(dir_name)
    models_pool["builtin_ai", None] = None

    elo_evaluator = ELOEvaluator(port=56789, use_bz2=True, tensorboard_dir="./log/elo_infos/tensorboard/")
    for model_id, weights in models_pool.items():
        elo_evaluator.add_model(model_id[0], model_id[1], weights=weights)
    elo_evaluator.run()


if __name__ == "__main__":
    main()
