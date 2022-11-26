import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# Reference implementation:
# https://github.com/albarji/deeprl-pong/blob/master/policygradientpytorch.py
# The inspiration for the neural net comes from this source as well as the
# choice of optimizer and hyperparameter.

# Works somewhat on pong, learns to return ball back quite well within 500,000 steps.
# Albeit it seems to get stuck in suboptimal local minima from time to time (maybe 1 in 3?)


class PGAgent(nn.Module):
    def __init__(self, actions, tb_writer, demo=False):
        super(PGAgent, self).__init__()
        n_actions = len(actions)
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2) , nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(start_dim=0), nn.Linear(1568, n_actions))
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.rewards = []
        self.grads = []
        self.eps = np.finfo(np.float32).eps.item()
        self.demo = demo
        if self.demo:
            y_pos = np.arange(n_actions)
            performance = [1]*n_actions
            plt.ion()
            self.figure, ax = plt.subplots(figsize=(6, 4))
            self.ln = ax.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, actions)
            plt.ylabel('Prob')
            plt.title('Action Probabilities')
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def act(self, observation, on_policy, demo=False):
        observation = (observation/255.0 - 0.5).astype(np.float32)
        observation_tensor = torch.tensor(observation).unsqueeze(0)
        x = self.net(observation_tensor)
        action_distribution = Categorical(F.softmax(x, dim=0))
        action = action_distribution.sample()
        self.net.zero_grad()
        loss = -action_distribution.log_prob(action)
        loss.backward()
        self.grad = [torch.clone(param.grad) for param in self.net.parameters()]
        if self.demo:
            for i in range(6):
                self.ln[i].set_height(action_distribution.probs[i].detach())
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        return action.item()

    def observe(self, observation, reward, done, reset):
        if done is False:
            self.rewards.append(reward)
            self.grads.append(self.grad)
        else:
            drewards = self.discount_rewards(self.rewards)
            length = len(self.grads)
            num_params = len(list(self.net.parameters()))
            weighted_grads = [[self.grads[exp_no][i] * drewards[exp_no] for exp_no in range(length)] for i in range(num_params)]
            weighted_grad = [torch.sum(torch.stack(weighted_grads[i]), dim=0) for i in range(num_params)]
            for p, i in zip(self.net.parameters(), range(num_params)):
                p.grad = weighted_grad[i]
            self.optimizer.step()
            self.grads = []
            self.rewards = []

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def discount_rewards(self, r, gamma=0.99):
        """ take 1D float array of rewards and compute discounted reward
    
        Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            # Actions preceeding reward affect that reward, but not subsequent rewards
            # Specific to pong
            if r[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + self.eps
        return discounted_r

    def episode_end(self, tb_writer):
        pass
