#Recurrent PFRL DeepQ
#Intentionally bit convoluted implementation for educational purpose (ie unwrap that RecurrentSequential module)

import pfrl
import torch
import torch.nn as nn
import gym
import numpy
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import agents, experiments, explorers
from pfrl.wrappers import atari_wrappers
import argparse
import logging
import sleep_wrapper

from pfrl.utils.recurrent import (
    get_packed_sequence_info,
    unwrap_packed_sequences_recursive,
    wrap_packed_sequences_recursive,
)


logging.basicConfig(level=20)

ap = argparse.ArgumentParser(description="pfrl RNN DQN")
ap.add_argument("--model")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--sleep", default=.02, type=float)
ap.add_argument("--env", default="PongNoFrameskip-v4")
ap.add_argument("--steps", default=1e7, type=int)
ns = ap.parse_args()

env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari(ns.env, max_frames=10000),
    episode_life=True,
    clip_rewards=True,
    frame_stack = False
)
test_env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari(ns.env, max_frames=10000),
    episode_life=False,
    clip_rewards=False,
    frame_stack = False
)


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv2d(1, 32, 8, stride=4)
        self.l2 = nn.ReLU()
        self.l3 = nn.Conv2d(32, 64, 4, stride=2)
        self.l4 = nn.ReLU()
        self.l5 = nn.Conv2d(64, 64, 3, stride=1)
        self.l6 = nn.Flatten()
        self.l7 = nn.ReLU()
        self.r8 = nn.LSTM(input_size=3136, hidden_size=512)
        self.l9 = init_chainer_default(torch.nn.Linear(512, n_actions))
        self.l10 = DiscreteActionValueHead()

    def forward(self, sequences, recurrent_state):
        h = sequences
        batch_sizes, sorted_indices = h.batch_sizes, h.sorted_indices
        if batch_sizes[0].item() == 32:
            debug = True
        else:
            debug = False
        h = h.data
        if debug:
            print("BS=", batch_sizes)
            print("H=", h.shape)
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)
        h = self.l7(h)
        if debug:
            print("H=", h.shape)
        h = torch.nn.utils.rnn.PackedSequence(
            h, batch_sizes=batch_sizes, sorted_indices=sorted_indices)
        if debug:
            print("Hdata=", h.data.shape)
        if recurrent_state is None:
            rs = None
        else:
            rs = recurrent_state[0]
        h, rs = self.r8(h, rs)
        new_recurrent_state = [rs]
        h = h.data
        h = self.l9(h)
        if debug:
            print("q=", h.shape)
        h = self.l10(h)
        h = wrap_packed_sequences_recursive(h, batch_sizes, sorted_indices)
        new_recurrent_state = [(new_recurrent_state[0][0]*0.0,new_recurrent_state[0][1]*0.0)]
        return h, tuple(new_recurrent_state)


q_func = MyNetwork()

replay_buffer = pfrl.replay_buffers.EpisodicReplayBuffer(10**6)

explorer = explorers.LinearDecayEpsilonGreedy(
    1.0,
    .01,
    ns.steps,
    lambda: numpy.random.randint(n_actions),
)

def phi(x):
    return numpy.asarray(x, dtype=numpy.float32) / 255

optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
    q_func.parameters(),
    lr=2.5e-4,
    alpha=0.95,
    momentum=0.0,
    eps=1e-2,
    centered=True,
)

agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma=0.99,
    explorer=explorer,
    replay_start_size=5*10**4,
#    replay_start_size=5*10**3,
    target_update_interval=10**4,
    clip_delta=True,
    update_interval=4,
    batch_accumulator="sum",
    phi=phi,
    gpu=0,
    episodic_update_len=4,
    recurrent=True
)

if ns.demo:
    agent.load(ns.model)
    test_env = pfrl.wrappers.Render(test_env)
    test_env = sleep_wrapper.SleepWrapper(test_env, ns.sleep)
    experiments.eval_performance(env=test_env, agent=agent, n_steps=100000, n_episodes=None)
else:
    experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=ns.steps,
        eval_n_steps=None,
        checkpoint_freq=None,
        eval_n_episodes=10,
        eval_interval=100000,
        outdir=ns.model,
        eval_env=test_env,
        use_tensorboard=True)
