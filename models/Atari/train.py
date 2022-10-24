import argparse
import gym
from pfrl.wrappers import atari_wrappers as pfrl_atari_wrappers
import pfrl_dqn
import py_dqn
import pg
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def experiment(env, agent, tb_writer=None, max_steps=500000):
    observation = env.reset()
    episode_score = 0
    episode_length = 0
    observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
    for idx in range(max_steps):
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        episode_score += reward
        episode_length += 1
        observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
        agent.observe(observation, reward, done, False)
        if done:
            observation = env.reset()
            observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
            print("Episode score=", episode_score, ", Length=", episode_length)
            if tb_writer is not None:
                tb_writer.add_scalar("episode_score", episode_score, idx)
                tb_writer.add_scalar("episode_length", episode_length, idx)
            episode_score = 0
            episode_length = 0


def demo(env, agent):
    observation = env.reset()
    episode_score = 0
    episode_length = 0
    observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
    for idx in range(500000):
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        episode_score += reward
        episode_length += 1
        observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
        if done:
            observation = env.reset()
            observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
            print("Episode score=", episode_score, ", Length=", episode_length)
            episode_score = 0
            episode_length = 0


ap = argparse.ArgumentParser(description="RL Trainer")
ap.add_argument("--env", default="PongNoFrameskip-v4")
ap.add_argument("--agent")
ap.add_argument("--max_steps", default=500000, type=int)
ap.add_argument("--folder", default=None)
#ap.add_argument("--device", default="cpu")
#ap.add_argument("--rollout_length", default=3, type=int)
ap.add_argument("--demo", action="store_true")
ns = ap.parse_args()

if ns.demo:
    render_mode = "human"
else:
    render_mode = "rgb_array"
env = gym.make(ns.env, render_mode=render_mode)
env.seed(42)
env = pfrl_atari_wrappers.MaxAndSkipEnv(env, skip=4)
env = pfrl_atari_wrappers.WarpFrame(env)
env = pfrl_atari_wrappers.FrameStack(env, 4)

tb_writer = None
if ns.folder is not None:
    tb_writer = SummaryWriter(ns.folder)

if ns.agent == "PFRLDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = pfrl_dqn.PFRLDQNAgent(tb_writer, q_max_steps)
elif ns.agent == "PG":
    agent = pg.PGAgent(tb_writer)
elif ns.agent == "PyDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = py_dqn.PyDQNAgent(tb_writer, q_max_steps)
else:
    print(ns.agent, " not recognised as agent.")
    quit()

if ns.demo:
    if ns.folder is not None:
        agent.load(ns.folder + "/model.pt")
    demo(env, agent)
else:
    experiment(env, agent, tb_writer, ns.max_steps)
    if ns.folder is not None:
        agent.save(ns.folder + "/model.pt")
