import random
import copy
import numpy as np
import q_model
import torch


class QLearnableStochasticModel:
    """Learns a probability distribution over MDP 's and stores a current MDP.

    Args:
    env: environment, is purely used to map states<->integers and retrieve environment width and height.

    State transitions are assumed stochastic, modelled as categorical distributions. Rewards, Dones are assumed
    deterministic, so once they have been observed, they are known.

    When sampling a new MDP:

    The state transition distributions are sampled from Dirichlet distributions where concentration parameters are
    counts of observed state transitions. Unknown rewards are assumed optimistic (+100). Unknown done's are bernoulli
    sampled with probability (0.5, 0.5)
    """
    # Inspiration:
    # Bayes Approach for Planning and Learning in Partially Observable Markov Decision Process,
    # Ross, Pineau, Chaib-Draa, Kreitmann, 2011
    def __init__(self, env):
        self.env = env  # Only used for the integer to state, state to integer functions
        self.height, self.width = self.env.height, self.env.width
        self.dones = np.zeros([self.height, self.width, 4]) - 1
        self.rewards = np.zeros([self.height, self.width, 4])
        self.num_states = self.height*self.width
        self.visits = np.zeros([self.height, self.width, 4], dtype=np.int64)
        self.state_transitions_dirichlet_alpha = np.ones([self.num_states, 4, self.num_states])*.01
        self.state_transition_cat_probs, self.random_rewards, self.random_dones = None, None, None
        self.sample_parameters()

    def sample(self, observation, action):
        """Sample a new state, reward, done from the currently sampled MDP given the observation and action."""
        done = self.dones[observation[0], observation[1], action]
        reward = self.rewards[observation[0], observation[1], action]
        observation_state = self.env.state_to_integer(observation)
        next_state = np.random.choice(self.num_states, p=self.state_transition_cat_probs[observation_state][action])
        if self.visits[observation[0], observation[1], action] == 0:
            reward = self.random_rewards[observation[0], observation[1], action]
            done = self.random_dones[observation[0], observation[1], action]
        new_observation = self.env.integer_to_state(next_state)
        return new_observation, reward, done, None

    def observe(self, observation, action, reward, done, next_observation):
        """Update the distribution over MDP's using this state transition.

        Note: does not affect current MDP"""
        self.dones[observation[0], observation[1], action] = done
        self.visits[observation[0], observation[1], action] = 1
        self.rewards[observation[0], observation[1], action] = reward
        observation_state = self.env.state_to_integer(observation)
        next_observation_state = self.env.state_to_integer(next_observation)
        self.state_transitions_dirichlet_alpha[observation_state, action, next_observation_state] += 1000

    def sample_parameters(self):
        """Resample a new MDP using the distribution over MDP's."""
        self.random_dones = np.random.binomial(1, p=np.ones([self.height, self.width, 4]) - .5)
        self.random_rewards = np.random.random([self.height, self.width, 4])+50
        self.state_transition_cat_probs =\
            [
                [
                    np.random.dirichlet(self.state_transitions_dirichlet_alpha[observation_state, action])
                    for action in range(4)
                ]
                for observation_state in range(self.num_states)
            ]

    def print_dones(self):
        print("Dones:")
        print("  Up")
        print("  ", self.dones[:, :, self.env.up])
        print("  Down")
        print("  ", self.dones[:, :, self.env.down])
        print("  Right")
        print("  ", self.dones[:, :, self.env.right])
        print("  Left")
        print("  ", self.dones[:, :, self.env.left])

    def print_visits(self):
        print("Visits:")
        print("  Up")
        print("  ", self.visits[:, :, self.env.up])
        print("  Down")
        print("  ", self.visits[:, :, self.env.down])
        print("  Right")
        print("  ", self.visits[:, :, self.env.right])
        print("  Left")
        print("  ", self.visits[:, :, self.env.left])

    def print_sampled_rewards(self):
        sampled_rewards = self.visits*self.rewards + (1-self.visits)*self.random_rewards
        print("Sampled Rewards:")
        print("  Up")
        print("  ", sampled_rewards[:, :, self.env.up])
        print("  Down")
        print("  ", sampled_rewards[:, :, self.env.down])
        print("  Right")
        print("  ", sampled_rewards[:, :, self.env.right])
        print("  Left")
        print("  ", sampled_rewards[:, :, self.env.left])

    def print_sampled_transitions(self):
        sampled_transitions = self.state_transition_cat_probs
        print("Sampled Transitions:")
        print("  Up")
        print("  ", sampled_transitions[:, :, self.env.up])
        print("  Down")
        print("  ", sampled_transitions[:, :, self.env.down])
        print("  Right")
        print("  ", sampled_transitions[:, :, self.env.right])
        print("  Left")
        print("  ", sampled_transitions[:, :, self.env.left])


class StochasticLearnableModelAgent:
    """Provides an agent interface.

    Learns a probability distribution over state transitions probabilities and deterministic reward and done functions.
    """
    def __init__(self, env):
        self.action = None
        self.observation = None
        self.learnable_model = QLearnableStochasticModel(env)
        self.q_algorithm = q_model.QAlgorithm(self.learnable_model, self.learnable_model.env)
        self.steps = 0

    def act(self, observation):
        """return best action from current MDP given current observation"""
        q = self.q_algorithm.q[observation[0], observation[1]]
        action = q.argmax()
        self.action = action
        self.observation = observation
        self.steps += 1
        return action

    def observe(self, observation, reward, done, _):
        """Update determistic functions reward and done, and update probability distribution over state transition
        probability distributions. Intermittently resample an MDP using the most optimistic (with respect to the
        current observation) of a sample of MDP's."""
        self.learnable_model.observe(self.observation, self.action, reward, done, observation)
        # Could consider replanning if new observations substantially change the probability distributions over MDP's,
        if self.steps % 20 == 0:
            print("Total explored=", self.learnable_model.visits.sum())
            best_q = -1000000
            best_mdp_state_transition_probs, best_mdp_random_rewards, best_mdp_random_dones = None, None, None
            for mdp_sample in range(10):
                self.learnable_model.sample_parameters()
                self.q_algorithm.reset()
                self.q_algorithm.update(planning_steps=120)
                if self.q_algorithm.q[self.observation[0], self.observation[1]].max() > best_q:
                    best_mdp_state_transition_probs = copy.deepcopy(self.learnable_model.state_transition_cat_probs)
                    best_mdp_random_rewards = copy.deepcopy(self.learnable_model.random_rewards)
                    best_mdp_random_dones = copy.deepcopy(self.learnable_model.random_dones)
            self.learnable_model.state_transition_cat_probs = best_mdp_state_transition_probs
            self.learnable_model.random_rewards = best_mdp_random_rewards
            self.learnable_model.random_dones = best_mdp_random_dones
