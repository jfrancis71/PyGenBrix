import random
import copy
import numpy as np
import q_model
import mdp_distribution


class LearnableModelAgent:
    def __init__(self, env):
        self.action = None
        self.observation = None
        self.mdp_distribution = mdp_distribution.DeterministicMDPDistribution(env.height*env.width)
        self.q_algorithm = q_model.QAlgorithm(self.mdp_distribution.sample_mdp())
        self.steps = 0
        self.env = env

    def act(self, observation):
        """return best action from current MDP given current observation"""
        state = self.env.state_to_integer(observation)
        q = self.q_algorithm.q[state]
        action = q.argmax()
        self.action = action
        self.observation = observation
        self.steps += 1
        return action

    def observe(self, observation, reward, done, _):
        """Update determistic functions reward and done, and update probability distribution over state transitions
        Intermittently resample an MDP using the most optimistic (with respect to the
        current observation) of a sample of MDP's."""
        state = self.env.state_to_integer(self.observation)
        next_state = self.env.state_to_integer(observation)
        self.mdp_distribution.update(state, self.action, reward, done, next_state)
        # Could consider replanning if new observations substantially change the probability distributions over MDP's,
        if self.steps % 20 == 0:
            print("Total explored=", self.mdp_distribution.visits.sum())
            best_q = -1000000
            best_mdp_state_transition_probs, best_mdp_random_rewards, best_mdp_random_dones = None, None, None
            for mdp_sample_i in range(10):
                mdp = self.mdp_distribution.sample_mdp()
                self.q_algorithm.mdp = mdp
                self.q_algorithm.reset()
                self.q_algorithm.update(planning_steps=120)
                if self.q_algorithm.q[next_state].max() > best_q:
                    best_mdp = copy.deepcopy(mdp)
                    best_q = self.q_algorithm.q[next_state].max()
            self.q_algorithm.mdp = best_mdp
            self.q_algorithm.reset()
            self.q_algorithm.update(planning_steps=120)