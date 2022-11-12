import argparse
import gym
from pfrl.wrappers import atari_wrappers as pfrl_atari_wrappers
import pfrl_dqn
import treebackup_dqn
import py_dqn
import py_tedqn
import pg
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def experiment(env, agent, tb_writer=None, max_steps=500000):
    steps = 0
    episodes = 0
    while steps < max_steps:
        episode_score, episode_length = experiment_episode(env, agent)
        print("Episode score=", episode_score, ", Length=", episode_length)
        if tb_writer is not None:
            tb_writer.add_scalar("episode_score", episode_score, steps)
            tb_writer.add_scalar("episode_length", episode_length, steps)
        steps += episode_length
        episodes += 1
        if episodes % 10 == 0:
            test_episode_score, test_episode_length = demo_episode(env, agent)
            print("Test episode score=", test_episode_score, ", Length=", test_episode_length)
            if tb_writer is not None:
                tb_writer.add_scalar("test_episode_score", test_episode_score, steps)
                tb_writer.add_scalar("test_episode_length", test_episode_length, steps)

def experiment_episode(env, agent):
    observation = env.reset()
    episode_score = 0
    episode_length = 0
    observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
    done = False
    while done is False:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        episode_score += reward
        episode_length += 1
        observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
        agent.observe(observation, reward, done, False)
    return episode_score, episode_length

def demo(env, agent):
    steps = 0
    while steps < 500000:
        episode_score, episode_length = demo_episode(env, agent)
        steps += episode_length

def demo_episode(env, agent):
    observation = env.reset()
    episode_score = 0
    episode_length = 0
    observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
    done = False
    while done is False:
        action = agent.act(observation, on_policy=True)
        observation, reward, done, info = env.step(action)
        episode_score += reward
        episode_length += 1
        observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
    return episode_score, episode_length


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

n_actions = env.action_space.n

tb_writer = None
if ns.folder is not None:
    tb_writer = SummaryWriter(ns.folder)

if ns.agent == "PFRLDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = pfrl_dqn.PFRLDQNAgent(n_actions, tb_writer, q_max_steps)
elif ns.agent == "PG":
    agent = pg.PGAgent(n_actions, tb_writer, ns.demo)
elif ns.agent == "PyDQN":
    agent = py_dqn.PyDQNAgent(n_actions, tb_writer, ns.max_steps)
elif ns.agent == "TreeBackupDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = treebackup_dqn.PyDQNAgent(n_actions, tb_writer, q_max_steps)
elif ns.agent == "PyTEDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = py_tedqn.PyTEDQNAgent(n_actions, tb_writer, q_max_steps)
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
