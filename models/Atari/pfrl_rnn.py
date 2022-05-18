#Achieved best score of 2.0

import pfrl
import torch
import torch.nn as nn
import gym
import numpy
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl import agents, experiments, explorers
from pfrl.wrappers import atari_wrappers
import argparse

import logging

logging.basicConfig(level=20)

env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari('PongNoFrameskip-v4', max_frames=10000),
    episode_life=True,
    clip_rewards=True,
    frame_stack = False
)

n_actions = 6
# Q-network with LSTM
q_func = pfrl.nn.RecurrentSequential(
    nn.Conv2d(1, 32, 8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, 4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, stride=1),
    nn.Flatten(),
    nn.ReLU(),
    nn.LSTM(input_size=3136, hidden_size=512),
    nn.Linear(512, n_actions),
    DiscreteActionValueHead(),
)
# Replay buffer that stores whole episodes
replay_buffer = pfrl.replay_buffers.EpisodicReplayBuffer(10**6)

explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

def phi(x):
    # Feature extractor
    return numpy.asarray(x, dtype=numpy.float32) / 255



# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-4, eps=1e-4)

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma=0.99,
    explorer=explorer,
    replay_start_size=10000,
    target_update_interval=100,
    update_interval=4,
    phi=phi,
    gpu=0,
    episodic_update_len=10,
    batch_accumulator="mean",
    minibatch_size=32,
    recurrent=True
)

ap = argparse.ArgumentParser(description="pfrl DQN")
ap.add_argument("--model")
ap.add_argument("--demo", action="store_true")
ns = ap.parse_args()

if ns.demo:
    agent.load(ns.model)
    env = pfrl.wrappers.Render(env)
    experiments.eval_performance(env=env, agent=agent, n_steps=100000, n_episodes=None)
else:
    experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=1000000,
        eval_n_steps=None,
        checkpoint_freq=None,
        eval_n_episodes=10,
        eval_interval=100000,
        outdir=ns.model,
        eval_env=env)
