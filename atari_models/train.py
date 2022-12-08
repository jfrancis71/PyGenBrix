import argparse
import gym
from pfrl.wrappers import atari_wrappers as pfrl_atari_wrappers
import pfrl_dqn
import treebackup_dqn
import py_dqn
import py_tedqn
import pg
import pg_eligibility_traces
import human_agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def experiment(env, agent, tb_writer=None, max_steps=500000):
    steps = 0
    episodes = 0
    while steps < max_steps:
        episode_score, episode_length = experiment_episode(env, agent, learn=True, on_policy=False)
        print("At step ", steps, ", episode score=", episode_score, ", Length=", episode_length)
        if tb_writer is not None:
            tb_writer.add_scalar("episode_score", episode_score, steps)
            tb_writer.add_scalar("episode_length", episode_length, steps)
        steps += episode_length
        episodes += 1
        if episodes % 10 == 0:
            test_episode_score, test_episode_length = experiment_episode(env, agent, learn=False, on_policy=True)
            print("Test episode score=", test_episode_score, ", Length=", test_episode_length)
            if tb_writer is not None:
                tb_writer.add_scalar("test_episode_score", test_episode_score, steps)
                tb_writer.add_scalar("test_episode_length", test_episode_length, steps)
                agent.episode_end(tb_writer)

def experiment_episode(env, agent, learn=False, on_policy=True):
    observation = env.reset()
    episode_score = 0
    episode_length = 0
    observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
    done = False
    while done is False:
        action = agent.act(observation, on_policy=demo)
        observation, reward, done, info = env.step(action)
        episode_score += reward
        episode_length += 1
        observation = np.moveaxis(observation, [0, 1, 2], [1, 2, 0])
        if learn:
            agent.observe(observation, reward, done, False)
    return episode_score, episode_length

def demo(env, agent):
    steps = 0
    while steps < 500000:
        episode_score, episode_length = experiment_episode(env, agent, learn=False, on_policy=True)
        steps += episode_length
        print("Episode score=", episode_score, ", Length=", episode_length)

ap = argparse.ArgumentParser(description="RL Trainer")
ap.add_argument("--agent")
ap.add_argument("--env", default="PongNoFrameskip-v4")
ap.add_argument("--max_steps", default=500000, type=int)
ap.add_argument("--multi_steps", default=2, type=int)
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

actions = env.env.get_action_meanings()
if ns.agent == "PFRLDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = pfrl_dqn.PFRLDQNAgent(actions, tb_writer, q_max_steps)
elif ns.agent == "PG":
    agent = pg.PGAgent(actions, tb_writer, ns.demo)
elif ns.agent == "PGEligibility":
    agent = pg_eligibility_traces.PGEligibilityTracesAgent(actions, tb_writer, ns.demo)
elif ns.agent == "PyDQN":
    agent = py_dqn.PyDQNAgent(actions, tb_writer, ns.max_steps)
elif ns.agent == "TreeBackupDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = treebackup_dqn.PyDQNAgent(actions, tb_writer, q_max_steps, ns.multi_steps)
elif ns.agent == "PyTEDQN":
    q_max_steps = ns.max_steps
    if ns.demo:
        q_max_steps = 0
    agent = py_tedqn.PyTEDQNAgent(actions, tb_writer, q_max_steps)
elif ns.agent == "HUMAN":
    agent = human_agent.HumanAgent()
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