import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99999
buffer_limit = 5000
batch_size = 128
random_steps = 0


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

    def __init__(self, sigma=0.1, is_relative_detach=True):
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

    def put(self, transition):
        self.buffer.append(transition)

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
    def __init__(self):
        super(Qnet, self).__init__()
        h = 64
        self.noise = GaussianNoise(sigma=0.05)
        # self.fc1 = nn.Linear(4, 128)
        self.fc1 = nn.Linear(4, h)
        # self.fc1 = nn.Linear(8, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)
        # self.fc4 = nn.Linear(h, 4)
        self.fc5 = nn.Linear(h, 2)

    def forward(self, x):
        x = self.fc1(x)
        # l1 = F.dropout(x, p=0.15)
        l1 = x
        # x = nl(self.fc2(x))
        x1 = torch.softmax(self.fc2(l1), dim=-1)
        x1 = self.noise(x1)
        # x1 = F.dropout(x1, p=0.15)

        x2 = torch.softmax(self.fc3(l1), dim=-1)
        x2 = self.noise(x2)
        # x2 = F.dropout(x2, p=0.15)

        x3 = torch.softmax(self.fc4(l1), dim=-1)
        x3 = self.noise(x3)
        # x3 = F.dropout(x3, p=0.15)

        x = self.fc5(x1 + x2 + x3)
        return x

    def sample_action(self, obs, epsilon, mem_size):
        coin = random.random()
        if coin < epsilon or mem_size < random_steps:
            # return random.randint(0, 3)
            return random.randint(0, 1)
        else:
            out = self.forward(obs)
            return out.argmax().item()


def train(q, q_target, memory, optimizer):

    losses = []
    if memory.size() < random_steps:
        return

    for i in range(800):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # n = torch.distributions.Uniform(0.99, 1.010).sample()

        # n = (torch.rand_like(s) - 0.5) * 0.5

        # s = s + n
        # s_prime = s_prime + n

        q_out = q(s)
        q_a = q_out.gather(1, a)
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        with torch.no_grad():

            max_q_prime = q(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        # loss = F.smooth_l1_loss(q_a, target)
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        # plot_grad_flow(q.named_parameters())
        optimizer.step()
        losses.append(loss.item())

    print(f"{torch.mean(torch.tensor(losses))}")


def main():
    import math

    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")
    q = Qnet()

    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find("Linear") != -1:
            # apply a uniform distribution to the weights and a bias=0
            # i = math.pi
            # i = 0.01
            # m.weight.data.uniform_(-i, i)
            # m.bias.data.uniform_(-i, i)

            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    q.apply(weights_init_uniform)
    # clip_value = 0.3
    # for p in q.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # q_target = Qnet()
    # q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 1
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(q.parameters(), lr=learning_rate, momentum=0.95)

    for n_epi in range(10000):

        # if n_epi % 10 == 0:
        #     print("reset opt")
        #     optimizer = optim.Adam(q.parameters(), lr=learning_rate)

        # epsilon = max(
        #     0.01, 0.06 - 0.01 * (n_epi / 50)
        # )  # Linear annealing from 8% to 1%

        epsilon = 0.0
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            s_prime, r, done, info = env.step(a)
            env.render()
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 400.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                train(q, None, memory, optimizer)

                # params = (
                #     (q.fc1, "weight"),
                #     (q.fc2, "weight"),
                #     (q.fc4, "weight"),
                # )

                # prune.global_unstructured(
                #     params, pruning_method=prune.L1Unstructured, amount=0.5
                # )

                break

        if n_epi % print_interval == 0 and n_epi != 0:
            # q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
