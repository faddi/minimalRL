import gym
import collections
import random

import torch
from torch import argmax
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

from typing import List
from torch.optim import optimizer

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


# import numpy as np

import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 1e-2
gamma = 0.999
buffer_limit = 10000
batch_size = 64
random_steps = 0
num_nets = 5


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.max_reward = 1.0

    def put(self, transition):
        self.buffer.append(transition)

    def set_max_reward(self, r):
        if self.max_reward is None:
            self.max_reward = r
        else:
            self.max_reward = max(self.max_reward, r)

    def sample(self, n):
        mini_batch = random.choices(self.buffer, k=n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


def nl(x):
    # return torch.log(torch.abs(x) + 0.1)
    # return torch.nn.functional.leaky_relu(x)
    # return torch.sigmoid(x)
    # return torch.tanh(x)
    return torch.relu(x)
    # return torch.sin(x)
    # return torch.relu(torch.fmod(x, 2))
    # return torch.tanh(torch.fmod(x, 2))


class Qnet(nn.Module):
    def __init__(self, env: gym.Env, memory: ReplayBuffer):
        super(Qnet, self).__init__()
        self.memory = memory
        h = 32
        self.noise = GaussianNoise(sigma=0.01, is_relative_detach=False)
        # self.fc1 = nn.Linear(4, h)
        self.inp = env.observation_space.shape[0]
        self.outp = env.action_space.n

        self.fc1 = nn.Linear(self.inp, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, self.outp)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon, mem_size):
        coin = random.random()
        if coin < epsilon or mem_size < random_steps:
            return random.randint(0, self.outp - 1)
            # return random.randint(0, 1)
        else:
            out = self.forward(obs.unsqueeze(0))
            return out.argmax().item()


def train(q, q_target, memory, optimizer, ep):
    losses = []
    if memory.size() < random_steps:
        return

    # bz = batch_size if memory.max_reward < 400 else batch_size * 8
    # bz = int(memory.max_reward) * 2
    bz = batch_size

    for i in range(400):
        s, a, r, s_prime, done_mask = memory.sample(bz)

        # n = torch.distributions.Uniform(0.99, 1.010).sample()

        # n = (torch.rand_like(s) - 0.5) * 0.5

        # s = s + n
        # s_prime = s_prime + n

        q_out = q(s)
        q_a = q_out.gather(1, a)  # / memory.max_reward
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        with torch.no_grad():

            # max_q_prime1 = q_target(s_prime).max(1)[0]
            # max_q_prime2 = q(s_prime).max(1)[0]
            # max_q = torch.stack((max_q_prime1, max_q_prime2), dim=1)
            # max_q_prime = max_q.min(1)[0].unsqueeze(1)

            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

            # max_q_prime1 = q_target(s_prime).max(1)[0].unsqueeze(1)
            # max_q_prime = q_target(s_prime).mean(1).unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

            target[target > 500.0] = torch.tensor(500.0)
            # target = torch.log(target + 1)
            # target = target / memory.max_reward

        # loss = F.smooth_l1_loss(q_a, target)
        # l = F.mse_loss(q_a, target, reduce=None)
        l = (q_a - target) ** 2
        # m = l.max()
        # l[l < 0.01] = 0
        loss = l.mean()
        # loss = torch.sigmoid(q_a - target).mean()

        optimizer.zero_grad()
        loss.backward()

        # q.fc1.weights

        # plot_grad_flow(q.named_parameters())
        # clip = 1.0 / (memory.max_reward ** 2)
        # for p in q.parameters():
        #     clip = torch.log(p.grad.abs().max() + 1)
        #     p.grad.data.clamp_(-clip, clip)

        # for p in q.parameters():
        #     if p.grad is None or len(p.grad.shape) == 1:
        #         continue

        #     if torch.randint(low=0, high=10, size=()) >= 5:
        #         r = torch.zeros_like(p.grad)
        #         p.grad *= r

        #     # idx = torch.argmax(p.grad, dim=1)
        #     # idx = torch.randint(0, p.grad.shape[1], (p.grad.shape[0], 1))

        #     # # p.grad.where(p.grad <= value, p.grad * 0.0001, p.grad)
        #     # r = torch.zeros_like(p.grad)
        #     # if torch.randint(low=0, high=10, size=()) >= 5:
        #     #     r[:, idx] = 1.0

        #     # p.grad *= r

        #     # for i, qq in enumerate(p.grad):
        #     #     if i == idx:
        #     #         qq *= 0.0001

        #     # half = len(p.grad) // 2
        #     # if ep % 10 > 5:
        #     #     p.grad[:half] *= 0.001  # or whatever other operation
        #     # else:
        #     #     p.grad[half:] *= 0.001  # or whatever other operation

        optimizer.step()
        # q.fc1.weight.data = torch.clamp(q.fc1.weight.data, -1.0, 1.0)
        # q.fc2.weight.data = torch.clamp(q.fc2.weight.data, -1.0, 1.0)
        losses.append(loss.item())

    writer.add_scalar("Loss/train", torch.mean(torch.tensor(losses)), ep)

    print(f"bz: {bz} loss: {torch.mean(torch.tensor(losses))}")


def main():
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")

    memories = [ReplayBuffer() for _ in range(num_nets)]
    qs = [Qnet(env, memories[i]) for i in range(num_nets)]

    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find("Linear") != -1:
            # apply a uniform distribution to the weights and a bias=0
            # i = math.pi
            i = 2.0
            m.weight.data.uniform_(-i, i)
            m.bias.data.uniform_(-i, i)

            # m.weight.data.fill_(1.0)
            # m.bias.data.fill_(0.0)

    # q.apply(weights_init_uniform)
    # q_target.apply(weights_init_uniform)
    # q_target.load_state_dict(q.state_dict())

    for i, q in enumerate(qs):
        if i == 0:
            continue

        q.load_state_dict(qs[0].state_dict())

    # clip_value = 0.3
    # for p in q.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # q_target = Qnet()
    # q_target.load_state_dict(q.state_dict())

    print_interval = 1
    score = 0.0

    optimizers = [
        optim.Adam(qs[i].parameters(), lr=learning_rate) for i in range(num_nets)
    ]
    step = 0

    for n_epi in range(10000):

        # if n_epi % 10 == 0:
        #     print("reset opt")
        #     optimizer = optim.Adam(q.parameters(), lr=learning_rate)

        # epsilon = max(
        #     0.01, 0.06 - 0.01 * (n_epi / 50)
        # )  # Linear annealing from 8% to 1%

        epsilon = 0.05
        s = env.reset()
        done = False

        while not done:
            # a = q.sample_action(, epsilon, memory.size())

            obs = torch.from_numpy(s).float().unsqueeze(0)

            outs = [q(obs).argmax().item() for q in qs]
            a = max(set(outs), key=outs.count)

            # outs = torch.stack([q(obs) for q in qs])
            # outs = torch.sum(outs, 0)

            # a = outs.argmax().item()

            s_prime, r, done, info = env.step(a)
            env.render()
            done_mask = 0.0 if done else 1.0

            data = (s, a, r / 500, s_prime, done_mask)
            # if done:
            #     data = (s, a, score / 500, s_prime, done_mask)
            # else:
            #     data = (s, a, 0.0, s_prime, done_mask)

            memories[step % len(memories)].put(data)

            s = s_prime

            score += r
            step += 1
            if done:
                [m.set_max_reward(score) for m in memories]
                writer.add_scalar("reward", score, n_epi)
                break

                # params = (
                #     (q.fc1, "weight"),
                #     (q.fc2, "weight"),
                #     (q.fc4, "weight"),
                # )

                # prune.global_unstructured(
                #     params, pruning_method=prune.L1Unstructured, amount=0.5
                # )

        if n_epi % print_interval == 0:
            # q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memories[0].size(), epsilon * 100
                )
            )
            score = 0.0

        for i in range(num_nets):
            q = qs[i]

            r = int(torch.randint(0, len(memories), ()).item())
            r2 = int(torch.randint(0, len(memories), ()).item())

            q_target = qs[r]
            memory = memories[r2]

            # q_target = qs[(i + 1) % len(qs)]
            # memory = memories[i]
            optimizer = optimizers[i]

            train(q, q_target, memory, optimizer, n_epi)

    env.close()


if __name__ == "__main__":
    main()
