import random
import copy
import numpy as np
import q_model
import mdp_distribution
import torch.nn as nn


class LearnableModelAgent(nn.Module):
    def __init__(self, mdp_distribution):
        super().__init__()
        self.action = None
        self.observation = None
        self.mdp_distribution = mdp_distribution
        self.q_algorithm = q_model.QAlgorithm(self.mdp_distribution.sample_mdp())
        self.steps = 0
        self.explored = -1

    def act(self, observation):
        """return best action from current MDP given current observation"""
        q = self.q_algorithm.q[observation]
        action = q.argmax()
        self.action = action
        self.observation = observation
        self.steps += 1
        return action.detach().cpu().numpy().item()

    def observe(self, observation, reward, done, _):
        """Update determistic functions reward and done, and update probability distribution over state transitions
        Intermittently resample an MDP using the most optimistic (with respect to the
        current observation) of a sample of MDP's."""
        state = self.observation
        next_state = observation
        self.mdp_distribution.update(state, self.action, reward, done, next_state)
        # Replanning if we have learnt something new.
        if self.steps % 20 == 0 or self.mdp_distribution.visits.sum() > self.explored:
            self.explored = self.mdp_distribution.visits.sum()
            best_q = -1000000
            best_mdp_state_transition_probs, best_mdp_random_rewards, best_mdp_random_dones = None, None, None
            device = next(self.parameters()).device
            for mdp_sample_i in range(10):
                mdp = self.mdp_distribution.sample_mdp().to(device)
                self.q_algorithm.mdp = mdp
                self.q_algorithm.reset()
                self.q_algorithm.update(planning_steps=120)
                if self.q_algorithm.q[next_state].max() > best_q:
                    best_mdp = copy.deepcopy(mdp)
                    best_q = self.q_algorithm.q[next_state].max()
            self.q_algorithm.mdp = best_mdp
            self.q_algorithm.reset()
            self.q_algorithm.update(planning_steps=120)
            print("Total explored=", self.explored, " best_q=", best_q.item())

    def to(self, device):
        self.mdp_distribution.to(device)
        self.q_algorithm = q_model.QAlgorithm(self.mdp_distribution.sample_mdp())
        self.q_algorithm.to(device)
        return self


class DeterministicLearnableModelAgent(LearnableModelAgent):
    def __init__(self, env):
        super().__init__(mdp_distribution.DeterministicMDPDistribution(env.observation_space.n, env.action_space.n))


class StochasticLearnableModelAgent(LearnableModelAgent):
    def __init__(self, env):
        super().__init__(mdp_distribution.StochasticMDPDistribution(env.observation_space.n, env.action_space.n))
