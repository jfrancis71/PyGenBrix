import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as Categorical


class SequenceHMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super().__init__()
        self.state_transitions = nn.Parameter(torch.randn(num_states, num_states))  # (state, state')
        self.observation_transitions = nn.Parameter(torch.randn(num_states, num_observations))  # (state, observation)
        self.prior_states = nn.Parameter(torch.randn([num_states]))  # (state)

    def log_prob(self, observations):
        # pad at end with 0's to mark end of sequence
        observation_sequence = torch.zeros(observations.shape[0], observations.shape[1]+1, dtype=torch.int64)
        observation_sequence[:, :-1] = observations
        alpha = Categorical(logits=self.observation_transitions).logits[:, observation_sequence[:, 0]].transpose(0, 1) +\
            Categorical(logits=self.prior_states).logits
        for observation_idx in range(1, observation_sequence.shape[1]):
            alpha = Categorical(logits=self.observation_transitions).logits[:, observation_sequence[:, observation_idx]].transpose(0, 1) + \
                torch.logsumexp(torch.transpose(Categorical(logits=self.state_transitions).logits, 0, 1) + alpha.unsqueeze(1), dim=2)
        return torch.logsumexp(alpha, dim=1)

    def sample(self):
        return self.sample_state_observations()[1]

    def sample_state_observations(self):
        state = Categorical(logits=self.prior_states).sample()
        state_sequence = [state]
        next_token = Categorical(logits=self.observation_transitions[state]).sample()
        observation_sequence = [next_token]
        while next_token != 0:
            state = Categorical(logits=self.state_transitions[state]).sample()
            next_token = Categorical(logits=self.observation_transitions[state]).sample()
            state_sequence.append(state)
            observation_sequence.append(next_token)
        return torch.tensor(state_sequence)[:-1], torch.tensor(observation_sequence)[:-1]