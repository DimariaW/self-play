from rl.league import ELOEvaluator
import random


def game(model_name, opponent_name):
    if model_name == "aver" or opponent_name == "aver":
        return random.choice([0, 1, 0.5])

    game_map = {
        ("rock", "rock"): 0.5,
        ("rock", "scissor"): 1,
        ("rock", "paper"): 0,
        ("scissor", "rock"): 0,
        ("scissor", "scissor"): 0.5,
        ("scissor", "paper"): 1,
        ("paper", "rock"): 1,
        ("paper", "scissor"): 0,
        ("paper", "paper"): 0.5
    }
    return game_map[model_name, opponent_name]


def main():
    elo_evaluator = ELOEvaluator()
    elo_evaluator.add_model("rock")
    elo_evaluator.add_model("scissor")
    elo_evaluator.add_model("paper")
    elo_evaluator.add_model("aver")

    for i in range(80):
        model_args, opponent_args = elo_evaluator.sample_models()
        infos = {
            "model_id": model_args[0],
            "opponent_id": opponent_args[0],
            "win": game(model_args[0][0], opponent_args[0][0])
        }
        elo_evaluator.update(infos)
        print(elo_evaluator.model_id_to_elo_score)

    print("end")


if __name__ == "__main__":
    main()
