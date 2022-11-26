# Credit: Below repo was very useful reference implementation:
# https://github.com/leonjovanovic/drl-dqn-atari-pong by Leon Jovanovic
# Following hyperparameters were inspired from that repo, and achieved good
# results on Pong in 500,000 steps.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pfrl
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.utils.batch_states import batch_states
from pfrl.replay_buffer import batch_experiences
from pfrl import agents, explorers
import py_dqn


def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

n_actions = 6


#Some ideas based on Thompson Sampling on an Ensemble of DQN's for exploration
class PyTEDQNAgent(nn.Module):
    def __init__(self, actions, tb_writer, max_steps):
        super(PyTEDQNAgent, self).__init__()
        n_actions = actions
        self.subagent1 = py_dqn.PyDQNAgent(n_actions, None, max_steps)
        self.subagent2 = py_dqn.PyDQNAgent(n_actions, None, max_steps)
        self.subagent1.moving_nn[-1].weight.data.fill_(0.0)
        self.subagent1.moving_nn[-1].bias.data.fill_(1.0)
        self.subagent1.target_nn[-1].weight.data.fill_(0.0)
        self.subagent1.target_nn[-1].bias.data.fill_(1.0)
        self.subagent2.moving_nn[-1].weight.data.fill_(0.0)
        self.subagent2.moving_nn[-1].bias.data.fill_(-1.0)
        self.subagent2.target_nn[-1].weight.data.fill_(0.0)
        self.subagent2.target_nn[-1].bias.data.fill_(-1.0)
        self.training_agent = 0
        self.steps = 0

    def act(self, observation, on_policy):
        self.observation = observation
        tensor_obs = torch.tensor(np.array([phi(observation)]), device="cuda")
        all_actions1 = self.subagent1.moving_nn(tensor_obs)[0]
        all_actions2 = self.subagent2.moving_nn(tensor_obs)[0]
        all_actions = torch.transpose(torch.stack([all_actions1, all_actions2]),0,1)
        mean = all_actions.mean(dim=1)
        std = all_actions.std(dim=1)
        all_actions = torch.normal(mean, std*.3).max(0).indices.item()#.3 arbitrary scaling factor
        self.action = all_actions
        if self.training_agent == 0:
            self.subagent1.action = self.action
            self.subagent1.observation = self.observation
        else:
            self.subagent2.action = self.action
            self.subagent2.observation = self.observation
        return self.action


    def observe(self, observation, reward, done, reset):
        if self.training_agent == 0:
            self.subagent1.observe(observation, reward, done, reset)
        else:
            self.subagent2.observe(observation, reward, done, reset)
        if self.steps % 1000 == 0:
            self.training_agent = (self.training_agent + 1) % 2
        self.steps += 1

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
