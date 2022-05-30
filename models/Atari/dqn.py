#1 Frame: 7.41, 15.1
#2 Frame: 18, 15.7
#4 Frame: 17.3, 17.2

import argparse

import gym
from stable_baselines3 import DQN
from pfrl.wrappers import atari_wrappers


ap = argparse.ArgumentParser(description="DQN")
ap.add_argument("--env", default="PongNoFrameskip-v4")
ap.add_argument("--frame_stacks", default=4)
ap.add_argument("--learning_starts", default=100000, type=int)
ap.add_argument("--total_timesteps", default=1000000, type=int)
ap.add_argument("--save_path")
ap.add_argument("--tensorboard_log")
ns = ap.parse_args()

env = atari_wrappers.wrap_deepmind(
    atari_wrappers.make_atari(ns.env, max_frames=10000),
    episode_life=True,
    clip_rewards=True,
)
model = DQN('CnnPolicy', env, verbose=1, buffer_size=10000, learning_rate=.0001, learning_starts=ns.learning_starts, target_update_interval=1000, tensorboard_log=ns.tensorboard_log)
model.learn(total_timesteps=ns.total_timesteps)
if ns.save_path is not None:
    model.save(ns.save_path)
