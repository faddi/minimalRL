import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# import numpy as np

import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 1e-2
gamma = 0.95
buffer_limit = 10000
batch_size = 64
random_steps = 0
num_nets = 3


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

    def clear(self):
        self.buffer.clear()

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


class Qnet(nn.Module):
    def __init__(self, env: gym.Env, memory: ReplayBuffer):
        super(Qnet, self).__init__()
        self.memory = memory
        h = 32
        self.noise = GaussianNoise(sigma=0.1, is_relative_detach=False)
        # self.fc1 = nn.Linear(4, h)
        self.inp = env.observation_space.shape[0]
        self.outp = env.action_space.n
        self.noise = GaussianNoise(0.05)

        self.fc1 = nn.Linear(self.inp, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, self.outp)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # x = self.noise(x)
        x = F.relu(self.fc2(x))
        # x = self.noise(x)
        x = self.fc3(x)  # * 500
        x = F.sigmoid(x) * 1 * 500  # self.memory.max_reward
        return x

    def sample_action(self, obs, epsilon, mem_size):
        coin = random.random()
        if coin < epsilon or mem_size < random_steps:
            return random.randint(0, self.outp - 1)
            # return random.randint(0, 1)
        else:
            out = self.forward(obs.unsqueeze(0))
            return out.argmax().item()


def train_(qs, memory, optimizers, ep):
    losses = []
    if memory.size() < random_steps:
        return

    bz = batch_size

    for i in range(800):
        target_index = torch.randint(0, len(qs), ())
        q = qs[target_index]
        optimizer = optimizers[target_index]

        s, a, r, s_prime, done_mask = memory.sample(bz)

        q_out = q(s)
        q_a = q_out.gather(1, a)  # / memory.max_reward
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        with torch.no_grad():

            q_min = qs[0](s_prime)

            for i in range(1, len(qs)):
                m = qs[i](s_prime)
                q_min = torch.min(m, q_min)

            qf = q_min

            # q_avg = qs[0](s_prime)

            # for i in range(1, len(qs)):
            #     m = qs[i](s_prime)
            #     q_avg += m

            # q_avg = q_avg / len(qs)

            max_q_prime = qf.max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

            # target[target > 500.0] = torch.tensor(500.0)
            # target = torch.log(target + 1)
            # target = target / memory.max_reward

        # loss = F.smooth_l1_loss(q_a, target)
        loss = F.mse_loss(q_a, target)
        # l = (q_a - target) ** 2
        # loss = l.mean()
        # loss = torch.sigmoid(q_a - target).mean()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

    writer.add_scalar("Loss/train", torch.mean(torch.tensor(losses)), ep)

    print(f"bz: {bz} loss: {torch.mean(torch.tensor(losses))}")


def main():
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")

    # memories = [ReplayBuffer() for _ in range(num_nets)]
    memory = ReplayBuffer()
    qs = [Qnet(env, memory) for i in range(num_nets)]

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

    # for i, q in enumerate(qs):
    #     if i == 0:
    #         continue

    #     q.load_state_dict(qs[0].state_dict())

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

            # outs = [q(obs).argmax().item() for q in qs]
            # a = max(set(outs), key=outs.count)

            coin = random.random()
            if coin < epsilon:
                a = random.randint(0, qs[0].outp - 1)
                # return random.randint(0, 1)
            else:

                m = qs[0](obs)
                for i in range(1, len(qs)):
                    m2 = qs[i](obs)
                    m = torch.max(m2, m)

                a = m.argmax().item()

                # o = qs[0](obs)
                # for i in range(1, len(qs)):
                #     o += qs[i](obs)

                # a = o.argmax().item()

            # outs = torch.stack([q(obs) for q in qs])
            # outs = torch.sum(outs, 0)

            # a = outs.argmax().item()

            s_prime, r, done, info = env.step(a)
            env.render()
            done_mask = 0.0 if done else 1.0

            data = (s, a, r, s_prime, done_mask)
            # if done:
            #     data = (s, a, score / 500, s_prime, done_mask)
            # else:
            #     data = (s, a, 0.0, s_prime, done_mask)

            # memories[step % len(memories)].put(data)
            memory.put(data)

            s = s_prime

            score += r
            step += 1
            if done:
                # [m.set_max_reward(score) for m in memories]
                memory.set_max_reward(score)
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
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0

        # if memory.size() > 100:
        train_(qs, memory, optimizers, n_epi)
        # memory.clear()

    env.close()


if __name__ == "__main__":
    main()
