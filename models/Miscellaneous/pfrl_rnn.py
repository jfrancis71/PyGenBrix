#Achieved best score of 2.0

import pfrl
import torch
import torch.nn as nn
import gym
from gym import spaces
import numpy
import random
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import agents, experiments, explorers
from pfrl.wrappers import atari_wrappers
import argparse
import logging

logging.basicConfig(level=20)

ap = argparse.ArgumentParser(description="pfrl RNN DQN")
ap.add_argument("--model")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--sleep", default=.02, type=float)
ap.add_argument("--two", action="store_true")
ns = ap.parse_args()

class MultiBanditEnv(gym.Env):
    def __init__(self):
        super(MultiBanditEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=[1], dtype=numpy.uint8)
        self.bandits = numpy.random.rand(5)
        self.n = 0

    def step(self, action):
        reward = numpy.random.binomial(1, self.bandits[action], 1)[0]
        observation = reward
        done = False
        if self.n > 100:
            done = True
        self.n += 1
        return numpy.array([observation]), reward, done, {}

    def reset(self):
        self.n = 0
        self.bandits = numpy.random.rand(5)
        return numpy.array([0.0], dtype=numpy.float32)

    def render(self, mode='human', close=False):
        pass


class TwoArmedBanditEnv(gym.Env):
    def __init__(self):
        super(TwoArmedBanditEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=[1], dtype=numpy.uint8)
        self.n = 0
        self.reset_bandits()
        self.render_list = []

    def step(self, action):
        if action == 0:
            reward = numpy.random.binomial(1, self.bandit1, 1)[0]
        else:
            reward = numpy.random.binomial(1, self.bandit2, 1)[0]
        observation = reward
        done = False
        if self.n > 100:
            done = True
        self.n += 1
        self.render_list = [" ", " ", " "] + [str(reward)]
        self.render_list[action] = "X"
        return numpy.array([observation]), reward, done, {}

    def reset(self):
        self.n = 0
        self.reset_bandits()
        return numpy.array([0.0], dtype=numpy.float32)

    def render(self, mode='human', close=False):
        print("".join(self.render_list))

    def reset_bandits(self):
        bandit = random.randint(0,1)
        if bandit == 0:
            self.bandit1 = 0.2
            self.bandit2 = 0.8
        else:
            self.bandit2 = 0.2
            self.bandit1 = 0.8


if ns.two:
    env = TwoArmedBanditEnv()
else:
    env = MultiBanditEnv()

n_actions = env.action_space.n
q_func = pfrl.nn.RecurrentSequential(
    nn.LSTM(input_size=1, hidden_size=512),
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
    return numpy.asarray(x, dtype=numpy.float32) / 1.0

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
