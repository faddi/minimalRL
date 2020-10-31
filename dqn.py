import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim

import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 64

env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
action_size = 2
state_size = 4


def nl(x):

    return torch.relu(x)


#     lt0 = x < 0
#     gte0 = x >= 0
#
#     x[lt0] = -torch.log(-x[lt0] + 1)
#     x[gte0] = torch.log(x[gte0] + 1)
#
#     return x


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
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        h = 256
        self.hidden = h
        self.state_embedding = nn.Sequential(
            nn.Linear(state_size, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()
        )

        self.output = nn.Sequential(nn.Linear(h, action_size))

    def forward(self, x):
        h1 = self.get_hidden(x)
        x = self.output(h1)
        return {"hidden": h1, "output": x}

    def get_hidden(self, x):
        return self.state_embedding(x)

    def from_hidden(self, s):
        return self.output(s)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)["output"]
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, action_size - 1)
        else:
            return out.argmax().item()

    def lookahead_action_old(self, obs, model):
        out = self.forward(obs)

        # 1 step
        model_out = model(
            out["hidden"].repeat(action_size, 1),
            torch.arange(action_size).view(action_size, -1).float(),
        )

        next_hidden = model_out["next_hidden"]

        next_qs = self.from_hidden(next_hidden)

        o = [0, 0]

        for i in range(out["output"].size(0)):
            for j in range(next_qs.size(0)):
                o[i] = max(o[i], out["output"][i] + next_qs[i, j])

        return np.argmax(o)

    def lookahead_action(self, obs, model):
        with torch.no_grad():
            out = self.forward(obs)

            o = out["output"]

            total = self.lookahead_nsteps(out["hidden"], model, 1)

            for i in range(action_size):
                o[i] += total[i]

            return np.argmax(o.detach().numpy())

    def lookahead_nsteps(self, hidden_state, model, step):
        o = [0 for x in range(action_size)]

        model_out = model(
            hidden_state.repeat(action_size, 1),
            torch.arange(action_size).view(action_size, -1).float(),
        )

        next_hidden = model_out["next_hidden"]

        next_qs = self.from_hidden(next_hidden)

        for i in range(action_size):  # action

            if step - 1 == 0:
                for j in range(action_size):
                    o[i] = max(o[i], next_qs[i, j])
            else:

                # [1, 3]
                iq = self.lookahead_nsteps(next_hidden[i], model, step - 1)
                for j in range(action_size):
                    o[i] = max(o[i], iq[j] + next_qs[i, j])

        return o


class Model(nn.Module):
    def __init__(self, input_size, action_size, reward_size=1):
        super(Model, self).__init__()
        self.input_size = input_size
        h = 256
        self.embed_state = nn.Linear(input_size, h)
        self.embed_action = nn.Linear(action_size, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, input_size)
        self.fc4 = nn.Linear(h, reward_size)

    def forward(self, state, action):
        state_embedding = nl(self.embed_state(state))
        action_embedding = nl(self.embed_action(action))

        x = state_embedding * action_embedding

        x = nl(self.fc2(x))
        next = self.fc3(x) + state
        reward = self.fc4(x)
        return {"next_hidden": next, "reward": reward}


def train_embedding(q, q_target, memory, optimizer):
    losses = []
    for i in range(1):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        with torch.no_grad():
            max_q_prime = q_target(s_prime)["output"].max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        q_out = q(s)["output"]
        q_a = q_out.gather(1, a)

        loss = F.smooth_l1_loss(q_a, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def train(q, q_target, model, memory, optimizer):
    losses = []
    for i in range(10):
        s, a_raw, r_raw, s_prime_raw, done_mask = memory.sample(batch_size)

        a = torch.randint(action_size, a_raw.shape).long()

        with torch.no_grad():
            h = q.get_hidden(s)
            out = model(h, a.float())

            s_prime = out["next_hidden"]
            r = out["reward"]
            max_q_prime = q_target.from_hidden(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        q_out = q(s)["output"]
        q_a = q_out.gather(1, a.long())

        loss = F.smooth_l1_loss(q_a, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def train_model(model, q, memory, optimizer):

    reward_losses = []
    state_losses = []
    total_losses = []

    for i in range(1):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        with torch.no_grad():
            hidden = q.get_hidden(s)
            target_hidden = q.get_hidden(s_prime)

        o = model(hidden.float(), a.float())

        state_loss = F.smooth_l1_loss(o["next_hidden"], target_hidden)
        reward_loss = F.smooth_l1_loss(o["reward"], r.float())

        loss = state_loss + reward_loss

        state_losses.append(state_loss.item())
        reward_losses.append(reward_loss.item())
        total_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(total_losses), np.mean(state_losses), np.mean(reward_losses)


def main():
    # env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")
    env = gym.make(env_name)
    seed_everything(env, 11)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    q_target.eval()

    model = Model(input_size=q.hidden, action_size=1)
    memory = ReplayBuffer()

    print_interval = 1
    score = 0.0
    optimizer_q_output = optim.Adam(q.output.parameters(), lr=learning_rate)
    optimizer_q_state = optim.Adam(q.state_embedding.parameters(), lr=learning_rate)
    optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    for n_epi in range(10000):
        epsilon = max(
            0.01, 0.01 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False
        while not done:
            step += 1
            env.render()
            # a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            a = q.lookahead_action(torch.from_numpy(s).float(), model)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            if memory.size() > batch_size:
                if step % 1 == 0:
                    l0 = train_embedding(q, q_target, memory, optimizer_q_state)
                    writer.add_scalar("Loss/embedding", l0, step)
                # l0 = 0
                l1, state_loss, reward_loss = train_model(
                    model, q, memory, optimizer_model
                )
                l2 = train(q, q_target, model, memory, optimizer_q_output)

                writer.add_scalar("Loss/model_total", l1, step)
                writer.add_scalar("Loss/model_state", state_loss, step)
                writer.add_scalar("Loss/model_reward", reward_loss, step)
                writer.add_scalar("Loss/q", l2, step)

                # if step % 10 == 0:
                #     print(f"model: {l1}, q_out: {l2}, q_embed: {l0}")
            if step < 1000 or step % 2000 == 0:
                q_target.load_state_dict(q.state_dict())

            score += r
            if done:
                writer.add_scalar("score", score, step)
                # print(
                #     f"model: {np.mean(ml1)}, q_out: {np.mean(ml2)}, q_embed: {np.mean(ml0)}"
                # )
                break

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
