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
        # sequence is used both for input and as a target for the LSTM network.
        # Therefore, it is prepended and appended with 0. The 0 prepend indicates
        # start of sequence, and the appended 0 indicates end of sequence.
        # The appropriate left and right subsequences are fed in as
        # input and target respectively
        sequence = torch.zeros([x.shape[0], x.shape[1] + 2], dtype=torch.int64)
        sequence[:, 1:-1] = x
        # I'm using 0 token as both start and end tokens; 0 token on input means start, 0 token on output means end
        logits = self(sequence)[0][:, :-1]
        log_prob = torch.sum(torch.distributions.categorical.Categorical(logits=logits).log_prob(sequence[:, 1:]), dim=1)
        return log_prob

    def sample(self):
        input_sequence = torch.tensor([[0]])
        next_token = [-1]
        while next_token[0] != 0:
            next_token_logits = self(input_sequence)[0][:, -1]
            next_token = torch.distributions.categorical.Categorical(logits=next_token_logits).sample()
            input_sequence = torch.cat((input_sequence, next_token.unsqueeze(1)), dim=1)
        sampled_sequence = input_sequence[0, 1:-1]  # remove start and end symbols
        return sampled_sequence

    def forward(self, x):
        embedded = self.emb_layer(x)
        lstm_output, hidden = self.lstm(embedded, None)
        out = self.fc(lstm_output)
        return out, hidden

