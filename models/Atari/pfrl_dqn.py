import numpy as np
import torch.nn as nn
import torch
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
import pfrl
from pfrl import agents, explorers


#Note you can use an env wrapper to do this conversion, but this way
#you avoid storing 64 bit or 32 bit floats in the replay buffer,
#more memory efficient to store bytes and do the neural net conversion later.
def phi(x):
    return np.asarray(x, dtype=np.float32) / 255


class PFRLDQNAgent(nn.Module):
    def __init__(self, n_actions, tb_writer, max_steps):
        super(PFRLDQNAgent, self).__init__()
        self.q_func = nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
        self.replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10**5)
        if max_steps == 0:
            self.explorer = explorers.Greedy()
        else:
            self.explorer = explorers.LinearDecayEpsilonGreedy(
                1.0, .01, max_steps,
                lambda: np.random.randint(n_actions),)
        self.optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
            self.q_func.parameters(),
            lr=2.5e-4,
            alpha=0.95,
            momentum=0.0,
            eps=1e-2,
            centered=True,)
        self.agent = pfrl.agents.DoubleDQN(
            self.q_func,
            self.optimizer,
            self.replay_buffer,
            gamma=0.99,
            explorer=self.explorer,
            replay_start_size=5*10**4,
            target_update_interval=10**4,
            clip_delta=True,
            update_interval=4,
            batch_accumulator="sum",
           phi=phi,
           gpu=0,)

    def act(self, observation, on_policy):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, reset):
        self.agent.observe(observation, reward, done, reset)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
