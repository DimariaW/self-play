import random
import time

import torch
from rl.algorithm import Algorithm


def generate_gae_data(i, shape):
    if i == 0:
        value = torch.randn(shape)
        reward = torch.randn(shape)
    elif i == 1:  # multi-agent
        value = torch.randn((*shape, 15))
        reward = torch.randn((*shape, 15))
    elif i == 2:  # multi-agent, multi-reward
        value = torch.randn((*shape, 5, 5))
        reward = torch.randn((*shape, 5, 5))
    else:
        value = torch.randn(*shape)
        reward = torch.randn(*shape)
    gamma = random.random()
    lbd = random.random()
    done = torch.randint(low=0, high=2, size=shape)
    done[:, -1] = 1
    bootstrap_mask = torch.randint(low=0, high=2, size=shape)
    done[bootstrap_mask == 0] = 1  # 被mask掉的地方done要为1
    return value, reward, done, bootstrap_mask, gamma, lbd


def test_gae(i, shape):
    value, reward, done, bootstrap_mask, gamma, lbd = generate_gae_data(i, shape)
    beg1 = time.time()
    a1, v1 = Algorithm.gae(value, reward, done, bootstrap_mask, gamma, lbd)
    beg2 = time.time()
    a2, v2 = Algorithm.gae_v2(value, reward, done, bootstrap_mask, gamma, lbd)
    beg3 = time.time()
    a3, v3 = Algorithm.gae_v3(value, reward, done, bootstrap_mask, gamma, lbd)
    beg4 = time.time()
    print(torch.sum(a1 - a2), torch.sum(a1 - a3), torch.sum(a2 - a3))
    print(torch.sum(v1 - v2), torch.sum(v1 - v3), torch.sum(v2 - v3))
    print(f"1 :{beg2 - beg1}, 2:{beg3 - beg2}, 3:{beg4 - beg3}")


def test_vtrace(i, shape):
    value, reward, done, bootstrap_mask, gamma, lbd = generate_gae_data(i, shape)
    rho = torch.ones(*value.shape)
    c = torch.ones(*value.shape)

    a1, v1 = Algorithm.gae(value, reward, done, bootstrap_mask, gamma, lbd)
    beg = time.time()
    vtrace_adv, vtrace_value = Algorithm.vtrace(value, reward, done, bootstrap_mask, gamma, lbd, rho, c)
    end = time.time()
    print(end-beg, torch.sum(vtrace_value-v1))


if __name__ == "__main__":
    test_gae(0, (640, 320))
    test_gae(1, (640, 320))
    test_gae(2, (640, 320))
    test_vtrace(0, (640, 320))
    test_vtrace(1, (640, 320))
    test_vtrace(2, (640, 320))



