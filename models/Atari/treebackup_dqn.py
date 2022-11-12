#Very basic tree backup implementation. Does 2 steps.
#Note as the target policy is a greedy policy, this usually will not even backup 2 steps, so
#not very different from standard DQN in practice.

import collections
import random
from collections import namedtuple
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


def phi(x):
    return np.asarray(x, dtype=np.float32) / 255


class PFRLReplayBuffer():
    def __init__(self):
        self.replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=15000, num_steps=2)

    def append(self, observation, action, reward, next_observation):
        self.replay_buffer.append(observation, action, reward, next_observation)

    def sample(self, batch_size):
        sample_buffer = self.replay_buffer.sample(batch_size)
        states = np.array([[transition["state"] for transition in batch] for batch in sample_buffer])
        actions = np.array([[transition["action"] for transition in batch] for batch in sample_buffer])
        rewards = np.array([[transition["reward"] for transition in batch] for batch in sample_buffer])
        next_states = np.array([[transition["next_state"] for transition in batch] for batch in sample_buffer])
        return states, actions, rewards, next_states


class ReplayBuffer():
    def __init__(self):
        self.states = collections.deque(maxlen=15000)
        self.actions = collections.deque(maxlen=15000)
        self.rewards = collections.deque(maxlen=15000)
        self.next_states = collections.deque(maxlen=15000)

    def append(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def sample(self, batch_size, multi_step):
        chosen_transitions = random.sample(range(0, len(self.states) - multi_step), batch_size)
        states = np.array([[self.states[t+s] for s in range(2)] for t in chosen_transitions])
        actions = np.array([[self.actions[t+s] for s in range(2)] for t in chosen_transitions])
        rewards = np.array([[self.rewards[t+s] for s in range(2)] for t in chosen_transitions])
        next_states = np.array([[self.next_states[t+s] for s in range(2)] for t in chosen_transitions])
        return states, actions, rewards, next_states


class PyDQNAgent(nn.Module):
    def __init__(self, actions, tb_writer, max_steps):
        super(PyDQNAgent, self).__init__()
        n_actions = len(actions)
#        self.replay_buffer = PFRLReplayBuffer()
        self.replay_buffer = ReplayBuffer()
        self.steps = 0
        self.target_update_interval = 1000
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

    def act(self, observation, on_policy):
        self.observation = observation
        self.action = self.explorer.select_action(self.steps, lambda:
            self.select_greedy_action(phi(observation)))
        return self.action

    def select_greedy_action(self, observation):
        tensor_obs = torch.tensor(np.array([observation]), device="cuda")
        all_actions = self.moving_nn(tensor_obs)
        return all_actions.max(1)[1].item()

    def observe(self, observation, reward, done, reset):
        self.replay_buffer.append(self.observation, self.action, reward, observation)       
        self.sample_and_improve(32)
        self.steps += 1

    def sample_and_improve(self, batch_size):
        if self.steps > 10001:
            sample_buffer = self.replay_buffer.sample(batch_size, 2)
            loss = self.loss(sample_buffer)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if ( self.steps % self.target_update_interval) == 0:
            self.target_nn.load_state_dict(self.moving_nn.state_dict())

    def state_value(self, states):
        return self.target_nn(states[:,0]).max(1)[0].detach()

    def treebackup(self, states, actions, rewards, next_states):
        pi_actions = self.target_nn(states[:,1]).max(1)[1].detach()
        pi_probs = (pi_actions == actions[:,1])*1.0
        s1 = self.target_nn(states[:,1]).max(1)[0].detach()
        backup = pi_probs * self.state_value(states[:,1:]) + (1-pi_probs) * ( rewards[:,1] + .99 * s1 )
        return backup


    def loss(self, exp_batch):
        states = torch.tensor(exp_batch[0], device="cuda", dtype=torch.float32)/255.0
        actions = torch.tensor(exp_batch[1], device="cuda")
        rewards = torch.tensor(exp_batch[2], device="cuda", dtype=torch.float32)
        next_states = torch.tensor(exp_batch[3], device="cuda", dtype=torch.float32)/255.0
        curr_state_action_value = self.moving_nn(states[:,0]).gather(1,actions[:,:1])[:,0]
        next_state_value = self.treebackup(states, actions, rewards, next_states)
        q_target = rewards[:,0] + .99 * next_state_value
        return nn.MSELoss()(curr_state_action_value, q_target)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
