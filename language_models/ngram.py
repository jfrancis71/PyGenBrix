# Some ideas on Hierarchical Priors, inspired by Wray Buntine, youtube: Dirichlet Processes

import itertools
import torch
import torchtext
import re

class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, multi_dataset):
        self.sentences = [(re.sub('[^A-Za-z0-9]+', ' ', sentence[0]).split()) for sentence in multi_dataset]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

class SentenceToIntDataset(torch.utils.data.Dataset):
    """This is paddable, ie the first word in the dictionary has a token_id of 1. A token_id of 0 indicates 
the end of sentence (to support padded sequences)"""
    """self.vocab[token_id] returns word associated with that token id, eg self.vocab[1] = 'Two'
    Note self.vocab[0] = None (this is the end of sentence padding token)
    self.vocab_dict["Two"] = 1
    """

    def __init__(self, sentence_dataset):
        self.sentence_dataset = sentence_dataset
        # need to retain unique list of words consistent across runs
        self.vocab = [None] + list(dict.fromkeys([word for sentence in self.sentence_dataset for word in sentence]
                                                 ).keys())
        self.vocab_size = len(self.vocab)
        self.vocab_dict = dict(zip(self.vocab, range(len(self.vocab))))
        self.int_dataset = [[self.vocab_dict[word] for word in sentence] for sentence in self.sentence_dataset]

    def __len__(self):
        return len(self.sentence_dataset)

    def __getitem__(self, idx):
        return self.int_dataset[idx]

    def convert(self, seq_token_ids):  # expects a torch vector
        seq_words = [self.vocab[seq_token_ids[word_indx]] for word_indx in range(seq_token_ids.shape[0])]
        return seq_words

multi_dataset = torchtext.datasets.Multi30k(split="train", language_pair=("en", "de"))

sentence_dataset = SentenceDataset(multi_dataset)

s = SentenceToIntDataset(sentence_dataset)

sentances = [ s[sentance_idx] + [0] for sentance_idx in range(len(s))]

word_sequence = torch.tensor(list(itertools.chain(*sentances)))

bigrams = word_sequence.unfold(0,2,1).tolist()

trigrams = word_sequence.unfold(0,3,1).tolist()

fourgrams = word_sequence.unfold(0,4,1).tolist()

def get_bigrams(seq):
    return torch.bincount(
        torch.tensor([bigrams[i][1] for i in range(len(bigrams)) if bigrams[i][:1] == seq], dtype=torch.long), minlength=len(s.vocab))

def get_trigrams(seq):
    return torch.bincount(
        torch.tensor([trigrams[i][2] for i in range(len(trigrams)) if trigrams[i][:2] == seq], dtype=torch.long), minlength=len(s.vocab))

def get_fourgrams(seq):
    return torch.bincount(
        torch.tensor([fourgrams[i][3] for i in range(len(fourgrams)) if fourgrams[i][:3] == seq], dtype=torch.long), minlength=len(s.vocab))

def sampler():
    token_list = []
    token = torch.distributions.categorical.Categorical(
        probs=torch.distributions.dirichlet.Dirichlet(1.0/len(s.vocab)+get_bigrams([0])).sample()).sample()
    token_list.append(token)
    token = torch.distributions.categorical.Categorical(
        probs=torch.distributions.dirichlet.Dirichlet(1.0/len(s.vocab)+get_bigrams(token_list[-1:])).sample()).sample()
    token_list.append(token)
    token = torch.distributions.categorical.Categorical(
        probs=torch.distributions.dirichlet.Dirichlet(1.0/len(s.vocab)+get_trigrams(token_list[-2:])).sample()).sample()
    token_list.append(token)
    while token_list[-1] != 0:
        print(token_list)
        prior = (1 / len(s.vocab)) + get_bigrams(token_list[-1:])
        prior = prior / prior.sum()
        prior =  prior + get_trigrams(token_list[-2:])
        prior = prior / prior.sum()
        token = torch.distributions.categorical.Categorical(
            probs=torch.distributions.dirichlet.Dirichlet(prior+get_fourgrams(token_list[-3:])).sample()).sample()
        token_list.append(token)
    return token_list

print([s.vocab[word.item()] for word in sampler()])




