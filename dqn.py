import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Hyperparameters
learning_rate = 0.0001
gamma = 0.98
buffer_limit = 50000
batch_size = 1


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, episode):
        self.buffer.append(episode)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)

        out = []

        for episode in mini_batch:

            out_epi = []

            for transition in episode:
                s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
                s, a, r, s_prime, done_mask = transition
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                done_mask_lst.append([done_mask])

                out_epi.append(
                    (
                        torch.tensor(s_lst, dtype=torch.float),
                        torch.tensor(a_lst),
                        torch.tensor(r_lst),
                        torch.tensor(s_prime_lst, dtype=torch.float),
                        torch.tensor(done_mask_lst),
                    )
                )

            out.append(out_epi)

        return out

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.h = 128
        self.fc1 = nn.Linear(4, self.h)
        self.fc3 = nn.Linear(self.h, 2)

        self.lstm = nn.RNN(self.h, self.h, 1)

    def forward(self, x, hs, cs):
        x = F.relu(self.fc1(x))
        s = x.shape
        x = x.view((batch_size, 1, self.h))
        # x, (hs, cs) = self.lstm(x, (hs, cs))
        x, (hs) = self.lstm(x, hs)
        x = x.view(s)
        x = self.fc3(x)

        return x, (hs, cs)

    def sample_action(self, obs, epsilon, hs, cs):
        out, state = self.forward(obs, hs, cs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1), state
        else:
            return out.argmax().item(), state

    def get_initial_state(self):
        hidden_state = torch.zeros(1, 1, self.h).detach()
        cell_state = torch.zeros(1, 1, self.h).detach()
        return (hidden_state, cell_state)


def train(q, q_target, memory, optimizer):
    losses = []
    for i in range(50):
        episodes = memory.sample(batch_size)

        # s, a, r, s_prime, done_mask = memory.sample(batch_size)

        for episode in episodes:
            state = q.get_initial_state()
            for (s, a, r, s_prime, done_mask) in episode:
                q_out, next_state = q(s, state[0].detach(), state[1].detach())
                q_a = q_out.gather(1, a)

                #
                with torch.no_grad():
                    qt, nn_state = q_target(s_prime, next_state[0], next_state[1])

                    max_q_prime = qt.max(1)[0].unsqueeze(1)
                    target = r + gamma * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)

                losses.append(loss.item())

                state = next_state

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print(f"loss: {np.mean(losses)}")


def main():
    env = gym.make("CartPole-v1")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 1
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False
        episode = []
        state = q.get_initial_state()

        while not done:
            a, state = q.sample_action(
                torch.from_numpy(s).float(), epsilon, state[0], state[1]
            )
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            episode.append((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                memory.put(episode)
                break

        if memory.size() > 0:
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
    main()
