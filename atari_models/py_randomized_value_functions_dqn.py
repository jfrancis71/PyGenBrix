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


# Some ideas based on youtube talk by Ian Osband, Deep Exploration via Randomized Value Functions
# Also see Deep Bootstrapped DQN by Osband, Blundell, Pritzel, Van Roy, 2016
# Also see Deep Exploration via Randomized Value Functions by Osband, Van Roy, Russo, Wen 2019 
# Have not really got it to work well. Maybe misunderstood
# On pong, the initial policy seems to igmore observations, so paddle is nearly always at edges,
# so interesting observations to learn from.
class PyRandomizedValueFunctionsDQNAgent(nn.Module):
    def __init__(self, actions, tb_writer, num_agents):
        super().__init__()
        n_actions = actions
        self.subagents = [py_dqn.PyDQNAgent(n_actions, None, 0) for agent_idx in range(num_agents)]
        self.training_agent = 0
        self.steps = 9900  # DQN's start training at 10,000; so allow for some entries in replay
                           # buffer before starting training.
        self.num_agents = num_agents

    def act(self, observation, on_policy):
        self.observation = observation
        tensor_obs = torch.tensor(np.array([phi(observation)]), device="cuda")
        all_actions_all_agents = [self.subagents[agent_idx].moving_nn(tensor_obs)[0] for agent_idx in range(self.num_agents)]
        if self.steps % 100 == 0:
            self.training_agent = np.random.randint(0, len(self.subagents))
        self.action = all_actions_all_agents[self.training_agent].argmax().item()
        return self.action

    def observe(self, observation, reward, done, reset):
        replacement_sampling = np.random.randint(0, len(self.subagents), len(self.subagents))
        for agent_idx in replacement_sampling:
            reward_noise = np.random.random()*0.1
            self.subagents[agent_idx].replay_buffer.append(self.observation, self.action, reward+reward_noise, observation)
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
