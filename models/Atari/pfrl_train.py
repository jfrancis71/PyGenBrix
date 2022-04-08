import pfrl
import torch
import torch.nn
import gym
import numpy
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl import agents, experiments, explorers
from pfrl.wrappers import atari_wrappers

import logging

logging.basicConfig(level=20)


#env = gym.make('PongNoFrameskip-v4')
env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari('PongNoFrameskip-v4', max_frames=10000),
            episode_life=not True,
            clip_rewards=not True,
        )
#env = pfrl.wrappers.Render(env)


# Set the discount factor that discounts future rewards.
gamma = 0.99

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 4)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
def phi(x):
    # Feature extractor
    return numpy.asarray(x, dtype=numpy.float32) / 255


# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = 0

n_actions = 6

q_func = torch.nn.Sequential(
            pnn.SmallAtariCNN(),
            init_chainer_default(torch.nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-4)

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=10000,
    update_interval=4,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)

experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=1000000,
        eval_n_steps=None,
        checkpoint_freq=None,
        eval_n_episodes=10,
        eval_interval=100000,
        outdir="fred",
        eval_env=env
)
