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


class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""
    def __init__(self, env, tensorboard_log=None):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2) , nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(start_dim=0), nn.Linear(1568, 6))
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.env = env
        if tensorboard_log is not None:
            self.writer = SummaryWriter(tensorboard_log)
        else:
            self.writer = None

    def forward(self, x):
        x = self.net(x)
        return F.softmax(x, dim=0)

    def get_action_distribution(self, observation):
        probs = self(observation)
        return Categorical(probs)

#Note, this assumes a vectorized environment, for use by evaluate_policy
    def predict(self, observation, state=None, deterministic=None):
        observation = np.transpose(observation[0]/255.0 - 0.5, (2,0,1)).astype(np.float32)
        observation_tensor = torch.tensor(observation).unsqueeze(0)
        action_distribution = self.get_action_distribution(observation_tensor)
        action = action_distribution.sample()
        return [action.item()], None
    
    def train_iter(self, episode):
        observations, rewards, actions, log_probs = self.collect_rollout()
        drewards = discount_rewards(rewards)
        sum_rewards = np.sum(rewards)
        policy_loss = [-log_prob * reward for log_prob, reward in zip(log_probs, drewards)]
        policy_loss = torch.stack(policy_loss).sum()
        print("Policy loss=", policy_loss, "rewards=", sum_rewards, "length=", len(actions))
        if self.writer is not None:
            self.writer.add_scalar("policy_loss", policy_loss, episode)
            self.writer.add_scalar("rewards", sum_rewards, episode)
            self.writer.add_scalar("length", len(actions), episode)
        self.optimizer.zero_grad()
        policy_loss.backward()
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

    def collect_rollout(self, steps=2000):
        observations, rewards, actions, log_probs = [], [], [], []
        observation = self.env.reset()
        for _ in range(steps):
            observation = np.transpose(observation/255.0 - 0.5, (2,0,1)).astype(np.float32)
            observation_tensor = torch.tensor(observation).unsqueeze(0)
            action_distribution = self.get_action_distribution(observation_tensor)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)
            new_observation, reward, done, info = self.env.step(action)
            observations.append(observation)
            rewards.append(reward)
            actions.append(action)
            log_probs.append(log_prob)
            if done:
                break
            observation = new_observation
        return observations, rewards, actions, log_probs

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
    ap = argparse.ArgumentParser(description="Pong PG")
    ap.add_argument("--max_episodes", default=1000, type=int)
    ap.add_argument("--save_filename")
    ap.add_argument("--tensorboard_log", default=None)
    ns = ap.parse_args()
    env = gym.make("PongNoFrameskip-v4")
    env.seed(42)
    env = sb3_wrappers.MaxAndSkipEnv(env, skip=4)
    env = sb3_wrappers.WarpFrame(env)
    env = wrappers.StackFramesWrapper(env, 1, np.uint8)
    model = Policy(env, tensorboard_log=ns.tensorboard_log)
    model.learn(ns.max_episodes)
    if ns.save_filename is not None:
        model.save(ns.save_filename)
