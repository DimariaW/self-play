import gym


def play():
    from gym.utils.play import play
    mapping = {
        (ord(" "),): 1,  # fire
        (ord("w"),): 2,  # right
        (ord("s"),): 3,  # left
        (ord("w"), ord(" ")): 4,  # right fire
        (ord("s"), ord(" ")): 5,  # left fire
    }
    env = gym.make("Pong-v4")
    env = gym.make("ALE/Qbert-v5")
    env = gym.make("ALE/Tennis-v5")
    play(env, keys_to_action=mapping, fps=30)


if __name__ == "__main__":
    play()