import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Hyperparameters
learning_rate = 0.001
gamma = 0.98
buffer_limit = 50000
batch_size = 128


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
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
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


def dtanh(x):
    ex = torch.exp(x)
    mex = torch.exp(-x)

    return 1 - ((ex - mex) ** 2 / (ex + mex) ** 2)


class Lin(nn.Module):
    def __init__(self, in_features, out_features, last=False):
        super(Lin, self).__init__()
        self.layer = nn.Linear(in_features, out_features, bias=True)
        # self.B = (torch.rand(2, out_features) - 0.5) / 2

        self.B = torch.randn(2, out_features) / 2
        self.layer.weight.zero_()
        self.layer.bias.zero_()
        self.last = last
        # self.Bb = torch.randn(1, out_features)

    def get_grad(self, errors):

        if not self.last:
            Berr = errors.mm(self.B)
            # Bout = Berr * dtanh(self.out)
            Bout = Berr * torch.cos(self.out)
            a = torch.transpose(Bout, 0, 1).mm(self.input)
            b = torch.cos(self.out)

        else:
            # last: (e) * (input)T
            # ee = torch.cat([errors, torch.zeros(8, 1)], 1)
            a = torch.transpose(errors, 0, 1).mm(self.input)
            b = errors

        return -a, -b

    def update_grad(self, errors):
        g, b = self.get_grad(errors)
        # self.layer.weight.grad = g
        self.layer.weight.grad = g
        self.layer.bias.grad = b.mean(-2)
        return 0

        # if self.layer.bias is not None:
        #     self.layer.bias.grad = g

    def forward(self, x):
        self.input = x
        x = self.layer(x)
        self.out = x

        return x


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        h = 128

        self.layers = nn.ModuleList()
        self.layers.append(Lin(4, h))

        for _ in range(2):
            self.layers.append(Lin(h, h))

        self.out = Lin(h, 2, last=True)

    def forward(self, x):

        for l in self.layers:
            x = torch.sin(l(x))

        out = torch.sigmoid(self.out(x)) * 500

        return out

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    losses = []
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # q_out = q(s)
        # q_a = q_out.gather(1, a)
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # target = r + gamma * max_q_prime * done_mask
        # loss = F.smooth_l1_loss(q_a, target, reduce=False)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_vals, max_indexes = q_target(s_prime).max(1)

        max_q_prime = max_vals.unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target, reduce=False)

        a = torch.zeros_like(q_out)
        a[:, max_indexes] = loss
        # e = torch.cat([errors, torch.zeros(8, 1)], 1)

        losses.append(loss.mean())

        optimizer.zero_grad()
        # loss.backward()
        for l in q.layers:
            l.update_grad(a)

        q.out.update_grad(a)

        optimizer.step()

    print(np.mean(losses))


def main():
    env = gym.make("CartPole-v1")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print(q)

    print_interval = 1
    score = 0.0
    # optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    optimizer = optim.SGD(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    with torch.no_grad():
        main()
