import random
import time

import torch
from rl.algorithm import Algorithm


def test_gae_vtrace_upgo():
    value = torch.randn((640, 1280))
    reward = torch.randn((640, 1280))
    gamma = random.random()
    lbd = random.random()
    done = torch.randint(low=0, high=2, size=(640, 1280))
    bootstrap_mask = torch.randint(low=0, high=2, size=(640, 1280))
    bootstrap_mask[:, -1] = 0
    done[bootstrap_mask == 0] = 1  # 被mask掉的地方done要为1

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

    rho = torch.ones(640, 1280)
    c = torch.ones(640, 1280)
    beg = time.time()
    vtrace_adv, vtrace_value = Algorithm.vtrace(value, reward, done, bootstrap_mask, gamma, lbd, rho, c)
    end = time.time()
    print(end-beg, torch.sum(vtrace_value-v1))

    beg = time.time()
    upgo_adv, upgo_value = Algorithm.upgo(value, reward, done, bootstrap_mask, gamma, lbd)
    end = time.time()
    print(end - beg)

    print(torch.sum(a1 * (1. - bootstrap_mask)),
          torch.sum(a2 * (1. - bootstrap_mask)),
          torch.sum(a3 * (1. - bootstrap_mask)),
          torch.sum(vtrace_adv * (1. - bootstrap_mask)),
          torch.sum(upgo_adv * (1. - bootstrap_mask)))

    print(torch.sum((v1 - value) * (1. - bootstrap_mask)),
          torch.sum((v2 - value) * (1. - bootstrap_mask)),
          torch.sum((v3 - value) * (1. - bootstrap_mask)),
          torch.sum((vtrace_value - value) * (1. - bootstrap_mask)),
          torch.sum((upgo_value - value) * (1. - bootstrap_mask)))

    rho = torch.rand(640, 1280, 3)
    c = torch.rand(640, 1280, 3)

    adv, value0 = Algorithm.vtrace_multi_action_head(value, reward, done, bootstrap_mask, gamma, lbd, rho=rho, c=c)
    adv1, value1 = Algorithm.vtrace(value, reward, done, bootstrap_mask, gamma, lbd, rho=rho[:, :, 0], c=c[:, :, 0])
    adv2, value2 = Algorithm.vtrace(value, reward, done, bootstrap_mask, gamma, lbd, rho=rho[:, :, 1], c=c[:, :, 1])
    adv3, value3 = Algorithm.vtrace(value, reward, done, bootstrap_mask, gamma, lbd, rho=rho[:, :, 2], c=c[:, :, 2])
    print(torch.sum(adv[:, :, 0] - adv1), torch.sum(adv[:, :, 1] - adv2), torch.sum(adv[:, :, 2] - adv3))
    print(torch.sum(value0[:, :, 0] - value1), torch.sum(value0[:, :, 1] - value2), torch.sum(value0[:, :, 2] - value3))


if __name__ == "__main__":
    test_gae_vtrace_upgo()


