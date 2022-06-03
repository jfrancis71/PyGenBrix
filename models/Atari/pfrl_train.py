#Achives good score generally (8,7,19,19)

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

ap = argparse.ArgumentParser(description="pfrl DQN")
ap.add_argument("--model")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--sleep", default=.02, type=float)
ap.add_argument("--env", default="PongNoFrameskip-v4")
ap.add_argument("--steps", default=1e7, type=int)
ns = ap.parse_args()

env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari(ns.env, max_frames=None),
    episode_life=True,
    clip_rewards=True,
)
test_env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari(ns.env, max_frames=None),
    episode_life=False,
    clip_rewards=False,
)

n_actions = test_env.action_space.n
q_func = torch.nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(torch.nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )

replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10**5)

explorer = explorers.LinearDecayEpsilonGreedy(
    1.0,
    .01,
    ns.steps,
    lambda: numpy.random.randint(n_actions),
)

#Note you can use an env wrapper to do this conversion, but this way
#you avoid storing 64 bit or 32 bit floats in the replay buffer,
#more memory efficient to store bytes and do the neural net conversion later.
def phi(x):
    return numpy.asarray(x, dtype=numpy.float32) / 255

#optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-4, eps=1e-4)
optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
    q_func.parameters(),
    lr=2.5e-4,
    alpha=0.95,
    momentum=0.0,
    eps=1e-2,
    centered=True,
)


agent = pfrl.agents.DQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma=0.99,
    explorer=explorer,
    replay_start_size=5*10**4,
    target_update_interval=10**4,
    clip_delta=True,
    update_interval=4,
    batch_accumulator="sum",
    phi=phi,
    gpu=0,
)

if ns.demo:
    agent.load(ns.model)
    env = pfrl.wrappers.Render(test_env)
    env = sleep_wrapper.SleepWrapper(env, ns.sleep)
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
