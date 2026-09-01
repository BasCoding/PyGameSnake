import random
from collections import deque

import numpy as np
import torch
from torch import nn


class Linear_QNet(nn.Module):
    """Small network mapping the 11-value state to a Q-value for each of the 3 actions."""

    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        return self


class QTrainer:
    """Performs a single Q-learning update using the Bellman equation."""

    def __init__(self, model, lr, gamma, device):
        self.model = model
        self.gamma = gamma
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array(action), dtype=torch.long, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device)

        # add a batch dimension when training on a single step
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # predicted Q-values for the current state
        pred = self.model(state)

        # target = reward, plus the discounted best future reward if the game continues
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        self.optimizer.step()


class Agent:
    """Holds the network, the replay memory, and the exploration/learning logic."""

    def __init__(self, device, lr=1e-3, gamma=0.9, max_memory=100_000, batch_size=1000):
        self.device = device
        self.n_games = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)
        self.model = Linear_QNet().to(device)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma, device=device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """Learn immediately from the single step just taken."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """Replay a random batch of past steps to reinforce what works."""
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        """
        Epsilon-greedy action selection: explore a lot early on, then increasingly
        trust the network. Epsilon shrinks as more games are played.
        """
        epsilon = max(5, 80 - self.n_games)
        action = [0, 0, 0]
        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
            move = torch.argmax(self.model(state0)).item()
        action[move] = 1
        return action
