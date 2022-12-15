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


# Some ideas based on talk by Ian Osband, Deep Exploration via Randomized Value Functions
class PyRandomizedValueFunctionsDQNAgent(nn.Module):
    def __init__(self, actions, tb_writer, max_steps):
        super().__init__()
        n_actions = actions
        self.subagent1 = py_dqn.PyDQNAgent(n_actions, None, max_steps)
        self.subagent2 = py_dqn.PyDQNAgent(n_actions, None, max_steps)
        self.training_agent = 0
        self.steps = 0

    def act(self, observation, on_policy):
        self.observation = observation
        tensor_obs = torch.tensor(np.array([phi(observation)]), device="cuda")
        all_actions1 = self.subagent1.moving_nn(tensor_obs)[0]
        all_actions2 = self.subagent2.moving_nn(tensor_obs)[0]
        if self.steps % 100 == 0:
            if all_actions1.max() > all_actions2.max():
                self.training_agent = 0
            else:
                self.training_agent = 1
        if self.training_agent == 0:
            self.action = all_actions1.argmax().item()
        else:
            self.action = all_actions2.argmax().item()
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
        self.steps += 1

    def episode_end(self, tb_writer):
        pass

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
