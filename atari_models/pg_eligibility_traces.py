import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import math
import py_utils


# Reference implementation:
# https://github.com/albarji/deeprl-pong/blob/master/policygradientpytorch.py
# The inspiration for the neural net comes from this source as well as the
# choice of optimizer and hyperparameter.

# Works somewhat on pong, learns to return ball back quite well within 500,000 steps.
# Albeit it seems to get stuck in suboptimal local minima from time to time (maybe 1 in 3?)


class PGEligibilityTracesAgent(nn.Module):
    def __init__(self, actions, tb_writer, demo=False):
        super(PGEligibilityTracesAgent, self).__init__()
        n_actions = len(actions)
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2) , nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(start_dim=0), nn.Linear(1568, n_actions))
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.z = [ torch.zeros_like(p) for p in self.net.parameters()]
        self.cumulative_grad = [ torch.zeros_like(p) for p in self.net.parameters()]
        self.demo = demo
        self.mean_reward = py_utils.MovingAverage(gamma=.9995)
        self.meansq_reward = py_utils.MovingAverage(gamma=.9995)
        if self.demo:
            y_pos = np.arange(6)
            performance = [1]*n_actions
            plt.ion()
            self.figure, ax = plt.subplots(figsize=(6, 4))
            self.ln = ax.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, actions)
            plt.ylabel('Prob')
            plt.title('Action Probabilities')
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        self.eps = np.finfo(np.float32).eps

    def act(self, observation, on_policy, demo=False):
        observation = (observation/255.0 - 0.5).astype(np.float32)
        observation_tensor = torch.tensor(observation).unsqueeze(0)
        x = self.net(observation_tensor)
        action_distribution = Categorical(F.softmax(x, dim=0))
        action = action_distribution.sample()
        self.net.zero_grad()
        loss = -action_distribution.log_prob(action)
        loss.backward()
        if self.demo:
            for i in range(6):
                self.ln[i].set_height(action_distribution.probs[i].detach())
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        return action.item()

    def observe(self, observation, reward, done, reset):
        self.mean_reward.update(reward)
        self.meansq_reward.update(reward**2)
        var_reward = self.meansq_reward.value() - self.mean_reward.value()**2
        std_reward = math.sqrt(var_reward)
        adj_reward = (reward-self.mean_reward.value())/(std_reward+self.eps)
        adj_reward = math.tanh(adj_reward*.25)
        for i, p in zip(range(len(self.z)), self.net.parameters()):
            self.z[i] = .99*self.z[i] + p.grad
            self.cumulative_grad[i] += adj_reward*self.z[i]
        # Actions preceeding reward affect that reward, but not subsequent rewards
        # Specific to pong
        if reward != 0:
            self.z = [ torch.zeros_like(p) for p in self.net.parameters()]
        if done is False:
            pass
        else:
            for i, p in zip(range(len(self.z)), self.net.parameters()):
                p.grad = self.cumulative_grad[i]
            self.optimizer.step()
            self.z = [ torch.zeros_like(p) for p in self.net.parameters()]
            self.cumulative_grad = [ torch.zeros_like(p) for p in self.net.parameters()]

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def episode_end(self, tb_writer):
        pass
