import numpy as np


class MDP:
    """Markov Decision Process - States are categorical variables with a stochastic state transition.
    rewards and dones are deterministic functions"""
    def __init__(self, num_states, num_actions, state_transition_probs, rewards, dones):
        if rewards.shape != (num_states, num_actions):
            raise RuntimeError("rewards shape {} does not match num_states {}".format(rewards.shape, num_states))
        if dones.shape != (num_states, num_actions):
            raise RuntimeError("dones shape {} does not match num_states {}".format(dones.shape, num_states))
        self.state_transition_probs = state_transition_probs
        self.rewards = rewards
        self.dones = dones
        self.num_states = num_states
        self.num_actions = num_actions

    def sample(self, state, action):
        """Sample a new state, reward, done given the observation and action."""
        done = self.dones[state, action]
        reward = self.rewards[state, action]
        next_state = np.random.choice(self.num_states, p=self.state_transition_probs[state][action])
        return next_state, reward, done, None
