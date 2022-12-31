import numpy as np
import mdp


class MDPDistribution:
    """Learns a probability distribution over MDP's.

    Args:
    num_states:

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
    # Also Ian Osband, YouTube: Deep Exploration via Randomized Value Functions
    # Seems to work well in this explicit finite state MDP world.
    def __init__(self, num_states, num_actions):
        self.dones = np.zeros([num_states, num_actions]) - 1
        self.rewards = np.zeros([num_states, num_actions])
        self.visits = np.zeros([num_states, num_actions], dtype=np.int64)
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = "cpu"

    def update(self, state, action, reward, done, next_state):
        """Update the distribution over MDP's using this state transition.

        Note: does not affect current MDP"""
        if self.visits[state, action] > 0:
            if self.rewards[state, action] != reward:
                raise RuntimeError("Reward is assumed deterministic, but current reward {} different from previous reward {} from state {} executing action {}".format(reward, self.rewards[state, action], state, action))
            if self.dones[state, action] != done:
                raise RuntimeError("Done is assumed deterministic, but current done {} different from previous done {} from state {} executing action {}".format(done, self.dones[state, action], state, action))
        self.rewards[state, action] = reward
        self.dones[state, action] = done
        self.visits[state, action] = 1

    def sample_mdp(self):
        rewards = (1-self.visits) * (np.random.random([self.num_states, self.num_actions])+50) + \
            self.visits * self.rewards
        dones = (1-self.visits) * (np.random.binomial(1, p=np.ones([self.num_states, self.num_actions]) - .5)) + \
            self.visits * self.dones
        return rewards, dones

    def print_dones(self):
        print("Dones:")
        print("  Up")
        print("  ", self.dones[:, self.env.up])
        print("  Down")
        print("  ", self.dones[:, self.env.down])
        print("  Right")
        print("  ", self.dones[:, self.env.right])
        print("  Left")
        print("  ", self.dones[:, self.env.left])

    def print_visits(self):
        print("Visits:")
        print("  Up")
        print("  ", self.visits[:, self.env.up])
        print("  Down")
        print("  ", self.visits[:, self.env.down])
        print("  Right")
        print("  ", self.visits[:, self.env.right])
        print("  Left")
        print("  ", self.visits[:, self.env.left])

    def print_sampled_rewards(self):
        sampled_rewards = self.visits*self.rewards + (1-self.visits)*self.random_rewards
        print("Sampled Rewards:")
        print("  Up")
        print("  ", sampled_rewards[:, self.env.up])
        print("  Down")
        print("  ", sampled_rewards[:, self.env.down])
        print("  Right")
        print("  ", sampled_rewards[:, self.env.right])
        print("  Left")
        print("  ", sampled_rewards[:, self.env.left])

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


class StochasticMDPDistribution(MDPDistribution):
    def __init__(self, num_states, num_actions):
        super(StochasticMDPDistribution, self).__init__(num_states, num_actions)
        self.state_transitions_dirichlet_alpha = np.ones([num_states, num_actions, num_states])*.01

    def update(self, state, action, reward, done, next_state):
        """Update the distribution over MDP's using this state transition.

        Note: does not affect current MDP"""
        super(StochasticMDPDistribution, self).update(state, action, reward, done, next_state)
        self.state_transitions_dirichlet_alpha[state, action, next_state] += 5000

    def sample_mdp(self):
        state_transition_cat_probs = np.array(
            [
                [
                    np.random.dirichlet(self.state_transitions_dirichlet_alpha[observation_state, action])
                    for action in range(self.num_actions)
                ]
                for observation_state in range(self.num_states)
            ])
        rewards, dones = super(StochasticMDPDistribution, self).sample_mdp()
        return mdp.MDP(self.num_states, self.num_actions, state_transition_cat_probs, rewards, dones, device=self.device)


class DeterministicMDPDistribution(MDPDistribution):
    def __init__(self, num_states, num_actions):
        super(DeterministicMDPDistribution, self).__init__(num_states, num_actions)
        self.state_transitions = np.zeros([self.num_states, self.num_actions, self.num_states])

    def update(self, state, action, reward, done, next_state):
        """Update the distribution over MDP's using this state transition.

        Note: does not affect current MDP"""
        if self.visits[state, action] > 0:
            if self.state_transitions[state, action, next_state] != 1.0:
                raise "State transition is assumed deterministic, but different value from previous update"
        self.state_transitions[state, action, next_state] = 1.0
        super(DeterministicMDPDistribution, self).update(state, action, reward, done, next_state)

    def sample_mdp(self):
        rewards, dones = super(DeterministicMDPDistribution, self).sample_mdp()
        random_transitions = np.random.randint(self.num_states, size=[self.num_states, self.num_actions])
        state_transitions = np.zeros([self.num_states, self.num_actions, self.num_states])
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if self.visits[s,a] == 0:
                    state_transitions[s, a, random_transitions[s, a]] = 1.0
                else:
                    state_transitions[s, a] = self.state_transitions[s,a]
        return mdp.MDP(self.num_states, self.num_actions, state_transitions, rewards, dones, device=self.device)

    def to(self, device):
        self.device = device
