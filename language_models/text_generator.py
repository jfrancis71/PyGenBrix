import torchtext
import torch.utils.data as torchdata
import torch
import re
import dist_layers.sequence_lstm as sequence_lstm


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, multi_dataset):
        self.sentences = [re.sub('[^A-Za-z0-9]+', ' ', sentence[0]).split() for sentence in multi_dataset]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class SentenceToIntDataset(torch.utils.data.Dataset):
    """This is paddable, ie the first word in the dictionary has a token_id of 1. A token_id of 0 indicates the end of sentence (to support padded sequences)"""
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


def collate(sentences):
    max_sentence_length = max([len(sentence) for sentence in sentences])
    padded_sentences = torch.zeros([len(sentences), max_sentence_length], dtype=torch.int64)
    for sentence_idx in range(len(sentences)):
        for word_idx in range(len(sentences[sentence_idx])):
            padded_sentences[sentence_idx, word_idx] = sentences[sentence_idx][word_idx]
    return padded_sentences


dataloader = torch.utils.data.DataLoader(s, collate_fn=collate, batch_size=16, shuffle=True)
sentence_net = sequence_lstm.SequenceLSTM(num_tokens=s.vocab_size)


def train():
    opt = torch.optim.Adam(sentence_net.parameters(), lr=.001)
    for e in range(2500):
        total_loss = 0.0
        print("Epoch ", e)
        for (batch_id, x) in enumerate(dataloader):
            sentence_net.zero_grad()
            loss = -torch.mean(sentence_net.log_prob(x))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print("Loss=", total_loss/len(s))
        sample_tokens_ids = sentence_net.sample()
        print("Sample tokens=", sample_tokens_ids)
        conv = s.convert(sample_tokens_ids)
        print("Conv=", conv)


train()
