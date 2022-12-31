import numpy as np
import torch
import torch.nn as nn


class MDP(nn.Module):
    """Markov Decision Process - States are categorical variables with a stochastic state transition.
    rewards and dones are deterministic functions"""
    def __init__(self, num_states, num_actions, state_transition_probs, rewards, dones, device="cpu"):
        super().__init__()
        if rewards.shape != (num_states, num_actions):
            raise RuntimeError("rewards shape {} does not match num_states {}".format(rewards.shape, num_states))
        if dones.shape != (num_states, num_actions):
            raise RuntimeError("dones shape {} does not match num_states {}".format(dones.shape, num_states))
        self.state_transition_probs = nn.Parameter(torch.tensor(state_transition_probs, device=device))
        self.rewards = nn.Parameter(torch.tensor(rewards, device=device))
        self.dones = nn.Parameter(torch.tensor(dones, device=device))
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_action_pair_transition_distribution = torch.distributions.categorical.Categorical(probs=torch.tensor(self.state_transition_probs, device=device))

    def sample(self, state, action):
        """Sample a new state, reward, done given the observation and action."""
        done = self.dones[state, action]
        reward = self.rewards[state, action]
        next_state = np.random.choice(self.num_states, p=self.state_transition_probs[state][action])
        return next_state, reward, done, None

    def samples(self):
        dones = self.dones
        rewards = self.rewards
        next_states = self.state_action_pair_transition_distribution.sample()
        return next_states, rewards, dones, None
