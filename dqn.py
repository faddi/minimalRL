import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.distributions.beta import Beta

import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 1e-2
gamma = 0.9
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


class Gumbel(nn.Module):
    """ 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    """

    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            r = (x >= 0).float()
            return r, r

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = (
                -torch.log(-torch.log(U1 + eps) + eps),
                -torch.log(-torch.log(U2 + eps) + eps),
            )
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard, soft


class GLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GLinear, self).__init__()
        self.gumbel = Gumbel()
        self.picker = nn.Linear(in_features, out_features)
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x, skip_gumbel=False):
        xl = self.layer(x)

        if skip_gumbel:
            out = xl
            p, p_soft = None, None
        else:
            p, p_soft = self.gumbel(self.picker(x))
            out = xl * p

        return {"out": out, "p": p, "p_soft": p_soft}


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        h = 32
        self.noise = GaussianNoise(sigma=0.05)
        self.fc1 = GLinear(4, h)
        self.fc2 = GLinear(h, h)
        self.fc3 = nn.Linear(h, 2)
        self.do = nn.Dropout()

    def forward(self, x, skip_gumbel=False):
        o1 = self.fc1(x, skip_gumbel)
        o2 = self.fc2(self.do(torch.tanh(o1["out"])), skip_gumbel)
        o3 = self.fc3(self.do(torch.tanh(o2["out"])))

        p = [o1["p"], o2["p"]]
        p_soft = [o1["p_soft"], o2["p_soft"]]

        return {"out": o3, "p": p, "p_soft": p_soft}

    def sample_action(self, obs, epsilon, mem_size, skip_gumbel=False):
        coin = random.random()
        if coin < epsilon or mem_size < random_steps:
            # return random.randint(0, 3)
            return random.randint(0, 1)
        else:
            out = self.forward(obs.unsqueeze(0), skip_gumbel)
            return out["out"].argmax().item()


# beta = Beta(torch.tensor([0.6]), torch.tensor([0.4]))
# beta.cdf()


def train(q, q_target, memory, optimizer, epoch, skip_gumbel=False):

    qt = q
    # qt = Qnet()
    # qt.load_state_dict(q.state_dict())
    # qt.eval()

    losses = []
    if memory.size() < random_steps:
        return

    for i in range(400):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # n = torch.distributions.Uniform(0.99, 1.010).sample()

        # n = (torch.rand_like(s) - 0.5) * 0.5

        # s = s + n
        # s_prime = s_prime + n

        q_out = q(s, skip_gumbel)
        d = q_out["p_soft"]
        # d = q_out["p"]
        q_a = q_out["out"].gather(1, a)
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        with torch.no_grad():
            # qp = q(s_prime)
            qp = qt(s_prime, skip_gumbel)

            max_q_prime = qp["out"].max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        ######

        # q_out = q(s)
        # q_a = q_out["out"].gather(1, a)
        # with torch.no_grad():
        #     qp = qt(s_prime)

        #     max_q_prime = qp["out"].max(1)[0].unsqueeze(1)
        #     target = max_q_prime * done_mask

        # loss = F.smooth_l1_loss(q_a - target, r)
        ######

        # loss = F.smooth_l1_loss(
        #     q_a, target
        # )  # + gateLoss / ( epoch + 1)  # / (torch.exp(-gateLoss))
        # loss = F.mse_loss(q_a, target)
        # if d[0] < 0.

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

    # q.apply(weights_init_uniform)
    # clip_value = 0.3
    # for p in q.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # q_target = Qnet()
    # q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 1
    score = 0.0
    # optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    o = []
    po = []

    for name, param in q.named_parameters():
        print(name)
        if "picker" in name:
            po.append(param)
        else:
            o.append(param)

    optimizer = optim.Adam(o, lr=learning_rate)
    picker_optimizer = optim.Adam(po, lr=learning_rate)

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
            use_gumbel = n_epi > 10
            a = q.sample_action(
                torch.from_numpy(s).float(), epsilon, memory.size(), use_gumbel
            )
            s_prime, r, done, info = env.step(a)
            env.render()
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                # if n_epi % 2 == 0:
                if (n_epi / 10) % 2 == 0:
                    # if not use_gumbel:
                    train(q, None, memory, optimizer, n_epi, True)
                else:
                    train(q, None, memory, picker_optimizer, n_epi)

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
