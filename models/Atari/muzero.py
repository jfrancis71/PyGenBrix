#Attempt at implementing muzero, only really implements the reward aspect.
#Achieves some success on pong after 1,000 episodes.

import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3.common.atari_wrappers as sb3_wrappers
import PyGenBrix.models.Atari.wrappers as wrappers
import random


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128+6,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU())
        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,128)
        self.l3 = nn.Linear(128,128)
        self.l4 = nn.Linear(128,128)

        self.p1 = nn.Linear(6,128)
        self.p2 = nn.Linear(6,128)
        self.p3 = nn.Linear(6,128)
        self.p4 = nn.Linear(6,128)

    def forward(self, hidden, action):
        one_hot = torch.nn.functional.one_hot(action, num_classes=6).to(dtype=torch.float32)
        x = hidden
        x = nn.functional.relu(self.l1(x) + self.p1(one_hot))
        x = nn.functional.relu(self.l2(x) + self.p2(one_hot))
        x = nn.functional.relu(self.l3(x) + self.p3(one_hot))
        return x


class Rollout(nn.Module):
    def __init__(self, length):
        super(Rollout, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2) , nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(start_dim=0), nn.Linear(1568, 128))
        self.rnn = RNN()
        self.reward = nn.Linear(128,1)
        self.rem_reward = nn.Linear(128,1)
        self.length = length

#Compute sum of reward 
    def forward(self, observation, actions):
        h = self.encoder(observation)
        rewards = []
        for _ in range(self.length):
            h = self.rnn(h, actions[0])
            rewards.append(self.reward(h))
            actions = actions[1:]
        rewards = torch.stack(rewards, dim=1)
        return rewards, self.rem_reward(h)


class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""
    def __init__(self, env, tensorboard_log=None, device="cpu", rollout_length=3):
        super(Policy, self).__init__()
        self.env = env
        if tensorboard_log is not None:
            self.writer = SummaryWriter(tensorboard_log)
        else:
            self.writer = None
        self.rollout = Rollout(rollout_length).to(device)
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.device = device

#Note, this assumes a vectorized environment, for use by evaluate_policy
    def predict(self, observation, state=None, deterministic=None):
        observation = np.transpose(observation[0]/255.0 - 0.5, (2,0,1)).astype(np.float32)
        observation_tensor = torch.tensor(observation, device=self.device).unsqueeze(0)
        action_distribution = self.get_action_distribution(observation_tensor)
        action = action_distribution.sample()
        return [action.item()], None
    
    def train_iter(self, episode):
        observations, rewards, actions = self.collect_rollout(episode=episode)
        losses = []
        for t in range(0,len(observations)-16):
            total_next_few_rewards = self.rollout(torch.unsqueeze(torch.tensor(observations[t], device=self.device), 0), actions[t:t+1+self.rollout.length])
            actual_next_few_rewards = np.sum(rewards[t:t+15])
            sim_rewards, sim_rem_wards = self.rollout(torch.unsqueeze(torch.tensor(observations[t], device=self.device), 0), actions[t:t+1+self.rollout.length])
            actual_rewards, actual_rem_rewards = (torch.tensor(rewards[t:t+self.rollout.length]).unsqueeze(0),
                torch.tensor([np.sum(rewards[t+self.rollout.length:t+15])]))
            diff_rewards, diff_rem_rewards = actual_rewards - sim_rewards, actual_rem_rewards - sim_rem_wards
#            diff_rewards = torch.tensor(actual_rewards) - sim_rewards
            losses.append(diff_rewards*diff_rewards + diff_rem_rewards*diff_rem_rewards)
        loss = torch.stack(losses).sum()
        if self.writer is not None:
            self.writer.add_scalar("loss", loss, episode)
            self.writer.add_scalar("length", len(observations), episode)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, max_episodes=1000):
        episode = 0
        while episode<max_episodes:
            print("Iteration", episode)
            self.train_iter(episode)
            episode += 1

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def value(self, hidden_state, k):
        value = 0.0
        if k == self.rollout.length:
            value = self.rollout.rem_reward(hidden_state)
        else:
            left_state = self.rollout.rnn(hidden_state, torch.tensor(2))
            right_state = self.rollout.rnn(hidden_state, torch.tensor(3))
            left_value = self.value(left_state, k+1) + self.rollout.reward(left_state)
            right_value = self.value(left_state, k+1) + self.rollout.reward(right_state)
            value = torch.max(left_value, right_value)
        return value

    def plan(self, observation):
        hidden_state = self.rollout.encoder(observation)
        hidden_state_u = self.rollout.rnn(hidden_state, torch.tensor(2))
        hidden_state_d = self.rollout.rnn(hidden_state, torch.tensor(3))
        u = self.value(hidden_state_u, 1)
        d = self.value(hidden_state_d, 1)
        if u > d:
            action = 2
        else:
            action = 3
        return action

    def collect_rollout(self, episode, steps=2000):
        observations, rewards, actions = [], [], []
        observation = self.env.reset()
        for _ in range(steps):
            observation = np.transpose(observation/255.0 - 0.5, (2,0,1)).astype(np.float32)
            observation_tensor = torch.tensor(observation, device=self.device).unsqueeze(0)
            action = -1
            if random.random() > episode/1000.0:
                action_distribution = Categorical(torch.tensor([1,1,1,1,1,1], device=self.device)/6.0)
                action = action_distribution.sample()
            else:
                action = torch.tensor(self.plan(observation_tensor), device=self.device)
            new_observation, reward, done, info = self.env.step(action)
            observations.append(observation)
            rewards.append(reward)
            actions.append(action)
            if done:
                break
            observation = new_observation
        return observations, rewards, actions

eps = np.finfo(np.float32).eps.item()

#Source for discount_rewards: https://github.com/albarji/deeprl-pong/blob/master/policygradientpytorch.py
def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward

    Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    return discounted_r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Policy Gradients")
    ap.add_argument("--env", default="PongNoFrameskip-v4")
    ap.add_argument("--max_episodes", default=1000, type=int)
    ap.add_argument("--save_filename")
    ap.add_argument("--tensorboard_log", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rollout_length", default=3, type=int)
    ns = ap.parse_args()
    env = gym.make(ns.env)
    env.seed(42)
    env = sb3_wrappers.MaxAndSkipEnv(env, skip=4)
    env = sb3_wrappers.WarpFrame(env)
    env = wrappers.StackFramesWrapper(env, 1, np.uint8)
    model = Policy(env, tensorboard_log=ns.tensorboard_log, device=ns.device, rollout_length=ns.rollout_length)
    model.learn(ns.max_episodes)
    if ns.save_filename is not None:
        model.save(ns.save_filename)
