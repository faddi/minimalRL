import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


def seed_everything(env, seed: int = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    env.seed(seed)


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


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        h = 32
        self.hidden = h
        self.fc1 = nn.Linear(4, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(h1))
        x = self.fc3(x)
        return {"hidden": h1, "output": x}

    def get_hidden(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

    def from_hidden(self, s):
        x = F.relu(self.fc2(s))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)["output"]
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


class Model(nn.Module):
    def __init__(self, input_size, action_size, reward_size=1):
        super(Model, self).__init__()
        self.input_size = input_size
        h = 32
        self.embed_state = nn.Linear(input_size, h)
        self.embed_action = nn.Linear(action_size, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, input_size)
        self.fc4 = nn.Linear(h, reward_size)

    def forward(self, state, action):
        state_embedding = F.relu(self.embed_state(state))
        action_embedding = F.relu(self.embed_action(action))

        x = state_embedding * action_embedding

        x = F.relu(self.fc2(x))
        next = self.fc3(x)
        reward = self.fc4(x)
        return {"next_hidden": next, "reward": reward}


def train(q, q_target, model, memory, optimizer):
    for i in range(1):
        s, a, r_raw, s_prime_raw, done_mask = memory.sample(batch_size)

        with torch.no_grad():
            h = q.get_hidden(s)
            out = model(h, a.float())

            s_prime = out["next_hidden"]
            r = out["reward"]
            max_q_prime = q_target.from_hidden(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        q_out = q(s)["output"]
        q_a = q_out.gather(1, a)

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_model(model, q, memory, optimizer):

    losses = []

    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        with torch.no_grad():
            hidden = q.get_hidden(s)
            target_hidden = q.get_hidden(s_prime)

        o = model(hidden, a.float())

        state_loss = F.smooth_l1_loss(o["next_hidden"], target_hidden)
        reward_loss = F.smooth_l1_loss(o["reward"], r)

        loss = state_loss + reward_loss

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def main():
    env = gym.make("CartPole-v1")
    seed_everything(env, 11)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    q_target.eval()

    model = Model(input_size=q.hidden, action_size=1)
    memory = ReplayBuffer()

    print_interval = 1
    score = 0.0
    optimizer_q = optim.Adam(q.parameters(), lr=learning_rate)
    optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)

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
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            if memory.size() > batch_size:
                train_model(model, q, memory, optimizer_model)
                train(q, q_target, model, memory, optimizer_q)

            score += r
            if done:
                break

        q_target.load_state_dict(q.state_dict())

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
