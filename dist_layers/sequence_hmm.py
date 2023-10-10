import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as Categorical


class HMM(nn.Module):
    def __init__(self, state_model, observation_model):
        super().__init__()
        self.observation_model = observation_model
        self.state_model = state_model
        self.current_state_distribution = self.state_model.prior_states()  # TODO as this is not for training, but arbitrary position start; should this be uniform probability over states?

    def log_prob(self, observations):
        alpha = self.observation_model.emission_logits(observations[:, 0]) +\
            self.state_model.prior_states()
        for observation_idx in range(1, observations.shape[1]):
            alpha = self.observation_model.emission_logits(observations[:, observation_idx]) + \
                torch.logsumexp(torch.transpose(self.state_model.state_transitions(), 0, 1) + alpha.unsqueeze(1), dim=2)
        return torch.logsumexp(alpha, dim=1)

    def observe(self, observation):
        alpha = self.observation_model.emission_logits(observation.unsqueeze(0))[0] + \
                torch.logsumexp(torch.transpose(self.state_model.state_transitions(), 0, 1) + self.current_state_distribution, dim=1)
        self.current_state_distribution = alpha - torch.logsumexp(alpha, dim=0)
        return alpha


class MatrixObservationModel(nn.Module):
    def __init__(self, num_states, num_observations):
        super().__init__()
        self.emission_logits_matrix = nn.Parameter(torch.randn(num_states, num_observations))  # (state, observation)

    def emission_logits(self, observation):  # (Batch, observation)
        log_prob = torch.distributions.categorical.Categorical(logits=self.emission_logits_matrix).logits
        """returns vector of length num_states with log p(states, observation)"""
        return log_prob[:, observation].transpose(0, 1)


class StateModel(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.prior_states_vector = nn.Parameter(torch.randn([num_states]))  # (state)
        self.state_transitions_matrix = nn.Parameter(torch.randn(num_states, num_states))  # (state, state')

    def prior_states(self):
        return self.prior_states_vector

    def state_transitions(self):
        return self.state_transitions_matrix

class SequenceHMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super().__init__()
        self.hmm = HMM(StateModel(num_states), MatrixObservationModel(num_states, num_observations))
        self.prior_states = self.hmm.state_model.prior_states_vector
        self.state_transitions = self.hmm.state_model.state_transitions_matrix
        self.emission_logits = self.hmm.observation_model.emission_logits_matrix

    def log_prob(self, observations):
        return self.hmm.log_prob(observations)


class TerminatingSequenceHMM(nn.Module):
    def __init__(self, num_states, num_observations):
        # num_observations includes the 0 token, ie language with only 1 word has num_observations=2
        super().__init__()
        self.hmm = SequenceHMM(num_states, num_observations)

    def log_prob(self, observations):
        # pad at end with 0's to mark end of sequence
        observation_sequence = torch.zeros(observations.shape[0], observations.shape[1]+1, dtype=torch.int64)
        observation_sequence[:, :-1] = observations
        return self.hmm.log_prob(observation_sequence)

    def sample(self):
        return self.sample_state_observations()[1]

    def sample_state_observations(self):
        state = Categorical(logits=self.hmm.prior_states).sample()
        state_sequence = [state]
        next_token = Categorical(logits=self.hmm.emission_logits[state]).sample()
        observation_sequence = [next_token]
        while next_token != 0:
            state = Categorical(logits=self.hmm.state_transitions[state]).sample()
            next_token = Categorical(logits=self.hmm.emission_logits[state]).sample()
            state_sequence.append(state)
            observation_sequence.append(next_token)
        return torch.tensor(state_sequence)[:-1], torch.tensor(observation_sequence)[:-1]
