import torchtext
import torch.utils.data as torchdata
import torch
import torch.nn as nn
import re
import argparse


class SentanceDataset(torch.utils.data.Dataset):
    def __init__(self, multi_dataset):
        self.sentances = [re.sub('[^A-Za-z0-9]+', ' ', sentance[0]).split() for sentance in multi_dataset]
        
    def __len__(self):
        return len(self.sentances)
    
    def __getitem__(self, idx):
        return self.sentances[idx]


class SentanceToIntDataset(torch.utils.data.Dataset):
    "Takes a dataset of list of sentance (list of strings) and converts to list of sentances where sentance is list of integers indexing into a vocabulary."
    "with a 0 and 1 prepended and appended respectively indicating start and end of sentance"
    def __init__(self, sentance_dataset):
        self.sentance_dataset = sentance_dataset
        #need to retain unique list of words consistent across runs
        self.vocab = list(dict.fromkeys([word for sentance in self.sentance_dataset for word in sentance]).keys())
        self.vocab_size = len(self.vocab)+2
        self.vocab_dict = dict(zip(self.vocab,range(2,len(self.vocab)+2)))
        self.int_dataset = [ [0] + [ self.vocab_dict[word] for word in sentance] + [1] for sentance in self.sentance_dataset]
        
    def __len__(self):
        return len(self.sentance_dataset)
    
    def __getitem__(self, idx):
        return self.int_dataset[idx]


class WordLSTM(nn.Module):
    def __init__(self, vocab_size, n_hidden=256, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden        
        self.emb_layer = nn.Embedding(vocab_size+2, 200)
        self.lstm = nn.LSTM(200, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, vocab_size+2)      
    
    def forward(self, x, hidden):
        embedded = self.emb_layer(x)     
        lstm_output, hidden = self.lstm(embedded, hidden)
        out = self.fc(lstm_output)
        return out, hidden


def train(sentance_net, int_dataset):
    opt = torch.optim.Adam(sentance_net.parameters(), lr=.001)
    for e in range(5):
        total_loss = 0.0
        print("Epoch ", e)
        for i in range(len(int_dataset)):
            sentance_net.zero_grad()
            logits = sentance_net(torch.tensor(int_dataset[i][:-1]).cuda(), None)[0]
            loss = -torch.sum(torch.distributions.categorical.Categorical(logits=logits).log_prob(torch.tensor(int_dataset[i][1:]).cuda()))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print("Loss=", total_loss/len(int_dataset))
        sample(sentance_net, int_dataset)


def sample(sentance_net, int_dataset):
    sentance = torch.tensor([0]).cuda()
    for i in range(35):
        logits, h = sentance_net(sentance, None)
        token = torch.tensor([torch.distributions.categorical.Categorical(logits=logits[i]).sample()]).cuda()
        sentance = torch.cat([sentance, token])
        if (token.item()) == 1:
            break
    print([ int_dataset.vocab[i-2] for i in sentance[1:-1] ])


multi_dataset = torchtext.datasets.Multi30k(split="train", language_pair=("en", "de"))
sentance_dataset = SentanceDataset(multi_dataset)
int_dataset = SentanceToIntDataset(sentance_dataset)
sentance_net = WordLSTM(int_dataset.vocab_size).cuda()

ap = argparse.ArgumentParser(description="text_generator")
ap.add_argument("--model")
ap.add_argument("--train", action="store_true")
ns = ap.parse_args()

if ns.train:
    train(sentance_net, int_dataset)
    torch.save(sentance_net.state_dict(), ns.model)
    sample(sentance_net, int_dataset)
else:
    sentance_net.load_state_dict(torch.load(ns.model))
    sample(sentance_net, int_dataset)
