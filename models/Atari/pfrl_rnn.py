#Achieved best score of 2.0

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

logging.basicConfig(level=20)

ap = argparse.ArgumentParser(description="pfrl RNN DQN")
ap.add_argument("--model")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--sleep", default=.02, type=float)
ap.add_argument("--env", default="PongNoFrameskip-v4")
ns = ap.parse_args()

env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari(ns.env, max_frames=10000),
    episode_life=True,
    clip_rewards=True,
    frame_stack = False
)

n_actions = env.action_space.n
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

replay_buffer = pfrl.replay_buffers.EpisodicReplayBuffer(10**6)

explorer = explorers.LinearDecayEpsilonGreedy(
    1.0,
    .01,
    1000000,
    lambda: numpy.random.randint(n_actions),
)

def phi(x):
    return numpy.asarray(x, dtype=numpy.float32) / 255

optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-4, eps=1e-4)

agent = pfrl.agents.DQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma=0.99,
    explorer=explorer,
    replay_start_size=10000,
    target_update_interval=1000,
    update_interval=1,
    phi=phi,
    gpu=0,
    episodic_update_len=10,
    recurrent=True
)

if ns.demo:
    agent.load(ns.model)
    env = pfrl.wrappers.Render(env)
    env = sleep_wrapper.SleepWrapper(env, ns.sleep)
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
        eval_env=env,
        use_tensorboard=True)
