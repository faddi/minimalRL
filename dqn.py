import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.01
gamma = 0.98
buffer_limit = 50000
batch_size = 32


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


class Lin(nn.Module):
    def __init__(self, in_features, out_features, loss_out_features):
        super(Lin, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.encoding = nn.Linear(out_features, loss_out_features)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def train_layer(self, q, target, memory, layer_trained):
        if self.training:
            train(q, target, memory, self.optimizer, layer_trained)

    def forward(self, x):
        x = F.relu(self.layer(x))
        encoding = self.encoding(x)
        return x, encoding


class Qnet(nn.Module):
    def __init__(self, memory):
        super(Qnet, self).__init__()
        self.fc1 = Lin(4, 128, 2)
        self.fc2 = Lin(128, 128, 2)
        self.fc3 = Lin(128, 2, 2)
        self.memory = memory

    def forward(self, x, target, train=True):
        x, e1 = self.fc1(x)
        if train:
            self.fc1.train_layer(self, target, self.memory, 0)
        x, e2 = self.fc2(x)
        if train:
            self.fc2.train_layer(self, target, self.memory, 1)
        x, e3 = self.fc3(x)
        if train:
            self.fc3.train_layer(self, target, self.memory, 2)
        return x, e1, e2, e3

    def sample_action(self, obs, epsilon, target):
        out = self.forward(obs, target)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out[0].argmax().item()


def train(q, q_target, memory, optimizer, layer_trained):

    if memory.size() < batch_size:
        return

    for i in range(1):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        e3, e0, e1, e2 = q(s, q_target, False)
        if layer_trained == 0:
            q_out = e0
        elif layer_trained == 1:
            q_out = e1
        elif layer_trained == 2:
            q_out = e2
        else:
            q_out = e3

        q_a = q_out.gather(1, a)
        with torch.no_grad():
            max_q_prime = q_target(s_prime, q_target, False)[0].max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    memory = ReplayBuffer()
    q = Qnet(memory)
    q_target = Qnet(memory)
    q_target.eval()
    q_target.load_state_dict(q.state_dict())

    print_interval = 1
    score = 0.0

    for n_epi in range(10000):
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, q_target)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        # if memory.size() > 2000:
        #     train(q, q_target, memory, optimizer)

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
    main()
