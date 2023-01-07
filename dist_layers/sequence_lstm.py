import torch
import torch.nn as nn


class SequenceLSTM(nn.Module):
    def __init__(self, num_tokens, n_hidden=256, n_layers=4):
        """num_tokens is number of tokens including any padding token (0)"""
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.emb_layer = nn.Embedding(num_tokens+1, 200)
        self.lstm = nn.LSTM(200, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, num_tokens)

    def log_prob(self, x):
        """t = torch.tensor([[1,2,3,4],[5,6,0,0]])
        seq.log_prob(t)
        >>> tensor([-11.3819, -11.3357], grad_fn=<SumBackward1>)
        """
        # x is a 0 padded sequence of shape [B,S]

        # sequence is used both for input and as a target for the LSTM network.
        # We prepend with 0 and appended with 1. The 0 prepend indicates
        # start of sequence, and the appended 1 indicates end of sequence.
        # The appropriate left and right subsequences are fed in as
        # input and target respectively. So note, the input to the LSTM can consist
        # of 0 and 1's. 0 indicating start of sequence, appended 1 indicating
        # end of sequence, and intermediate 1's padding sequence to same length
        sequence = torch.zeros([x.shape[0], x.shape[1] + 2], dtype=torch.int64)
        sequence[:, 1:-1] = x+1
        sequence[:, -1] = 1
        logits = self(sequence)[0][:, :-1]
        log_prob = torch.sum(torch.distributions.categorical.Categorical(logits=logits).log_prob(sequence[:, 1:]-1), dim=1)
        return log_prob

    def sample(self):
        input_sequence = torch.tensor([[0]])
        next_token = [-1]
        while next_token[0] != 0:
            next_token_logits = self(input_sequence)[0][:, -1]
            next_token = torch.distributions.categorical.Categorical(logits=next_token_logits).sample()
            input_sequence = torch.cat((input_sequence, 1+next_token.unsqueeze(1)), dim=1)
        sampled_sequence = input_sequence[0, 1:-1]-1  # remove start and end symbols
        return sampled_sequence

    def forward(self, x):
        embedded = self.emb_layer(x)
        lstm_output, hidden = self.lstm(embedded, None)
        out = self.fc(lstm_output)
        return out, hidden

