# Credit: This repo was very useful reference implementation:
# https://github.com/leonjovanovic/drl-dqn-atari-pong
# from Leon Jovanovic


import numpy as np
import torch.nn as nn
import torch
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.utils.batch_states import batch_states
from pfrl.replay_buffer import batch_experiences
import pfrl
from pfrl import agents, explorers
import torch.nn.functional as F
from numpy import random as nprandom
import random
from neural_nets import DQN
import torch.optim as optim


def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

n_actions = 6


class PyDQNAgent(nn.Module):
    def __init__(self, tb_writer, max_steps):
        super(PyDQNAgent, self).__init__()
        self.replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=15000)
        self.num_iterations = 0
        self.n_iter_update_nn = 1000
        if max_steps == 0:
            self.explorer = explorers.Greedy()
        else:
            self.explorer = explorers.LinearDecayEpsilonGreedy(
                1.0, .02, 10 ** 5, #note difference in exploration steps
                lambda: np.random.randint(n_actions),)
        self.moving_nn = nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            ).to("cuda")
        self.target_nn = nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            ).to("cuda")
        self.optimizer = optim.RMSprop(self.moving_nn.parameters(), lr=5e-5)

    def act(self, observation):
        self.observation = observation
        self.action = self.explorer.select_action(self.num_iterations, lambda:
            self.select_greedy_action(phi(observation)))
        return self.action

    def select_greedy_action(self, observation):
        tensor_obs = torch.tensor(np.array([observation]), device="cuda")
        all_actions = self.moving_nn(tensor_obs)
        return all_actions.max(1)[1].item()

    def observe(self, observation, reward, done, reset):
        self.replay_buffer.append(self.observation, self.action, reward, observation)       
        self.sample_and_improve(32)
        self.num_iterations += 1

    def sample_and_improve(self, batch_size):
        if self.num_iterations > 10001:
            sample_buffer = self.replay_buffer.sample(batch_size)
            exp_batch = batch_experiences(
                sample_buffer,
                device="cuda",
                phi=phi,
                gamma=.99,
                batch_states=batch_states,
            )
            loss = self.loss(exp_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if ( self.num_iterations % self.n_iter_update_nn) == 0:
            self.target_nn.load_state_dict(self.moving_nn.state_dict())

    def loss(self, exp_batch):
        curr_state_action_value = self.moving_nn(exp_batch["state"]).gather(1,exp_batch["action"][:,None]).squeeze(-1)
        next_state_action_value = self.target_nn(exp_batch["next_state"]).max(1)[0].detach()
        q_target = exp_batch["reward"] + .99 * next_state_action_value
        return nn.MSELoss()(curr_state_action_value, q_target)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
