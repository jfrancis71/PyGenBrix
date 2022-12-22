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
# Also see Bootstrapped DQN by Ian Osband
# Have not really got it to work well. Maybe misunderstood
class PyRandomizedValueFunctionsDQNAgent(nn.Module):
    def __init__(self, actions, tb_writer, num_agents):
        super().__init__()
        n_actions = actions
        self.subagents = [py_dqn.PyDQNAgent(n_actions, None, 0) for agent_idx in range(num_agents)]
        self.replay_buffer = py_dqn.ReplayBuffer()
        for agent_idx in range(num_agents):
            self.subagents[agent_idx].replay_buffer = self.replay_buffer
        self.training_agent = 0
        self.steps = 0
        self.num_agents = num_agents

    def act(self, observation, on_policy):
        self.observation = observation
        tensor_obs = torch.tensor(np.array([phi(observation)]), device="cuda")
        all_actions_all_agents = [self.subagents[agent_idx].moving_nn(tensor_obs)[0] for agent_idx in range(self.num_agents)]
        if self.steps % 100 == 0:
            best_action_all_agents = [all_actions_all_agents[agent_idx].max().detach().cpu() for agent_idx in range(self.num_agents)]
            self.training_agent = np.array(best_action_all_agents).argmax().item()
        self.action = all_actions_all_agents[self.training_agent].argmax().item()
        return self.action

    def observe(self, observation, reward, done, reset):
        self.replay_buffer.append(self.observation, self.action, reward, observation)
        for agent_idx in range(self.num_agents):
            self.subagents[agent_idx].steps = self.steps
            self.subagents[agent_idx].sample_and_improve(32)
        self.steps += 1

    def episode_end(self, tb_writer):
        pass

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
