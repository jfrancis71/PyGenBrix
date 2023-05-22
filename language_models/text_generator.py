# For LSTM achieves around 18.9, and HMM around 54 by epoch 46


import argparse
import torchtext
import torch.utils.data as torchdata
import torch
import re
import PyGenBrix.dist_layers.sequence_lstm as sequence_lstm
import PyGenBrix.dist_layers.sequence_hmm as sequence_hmm
import PyGenBrix.utils.py_train as pytrain


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, multi_dataset):
        self.sentences = [(re.sub('[^A-Za-z0-9]+', ' ', sentence[0]).split()) for sentence in multi_dataset]

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


def collate(sentences):
    max_sentence_length = max([len(sentence) for sentence in sentences])
    padded_sentences = torch.zeros([len(sentences), max_sentence_length], dtype=torch.int64)
    for sentence_idx in range(len(sentences)):
        for word_idx in range(len(sentences[sentence_idx])):
            padded_sentences[sentence_idx, word_idx] = sentences[sentence_idx][word_idx]
    return padded_sentences


def epoch_end_callback(distribution, epoch_num, loss):
    print("Epoch ", epoch_num, " loss=", loss)
    sample_tokens_ids = sentence_net.sample()
    print("Sample tokens=", sample_tokens_ids)
    conv = s.convert(sample_tokens_ids)
    print("Conv=", conv)



multi_dataset = torchtext.datasets.Multi30k(split="train", language_pair=("en", "de"))
sentence_dataset = SentenceDataset(multi_dataset)
s = SentenceToIntDataset(sentence_dataset)
dataloader = torch.utils.data.DataLoader(s, collate_fn=collate, batch_size=16, shuffle=True)

ap = argparse.ArgumentParser(description="Text Generator")
ap.add_argument("--model")
ns = ap.parse_args()

if ns.model == "lstm":
    sentence_net = sequence_lstm.SequenceLSTM(s.vocab_size)
elif ns.model == "hmm":
    sentence_net = sequence_hmm.TerminatingSequenceHMM(num_states=128, num_observations=s.vocab_size)
else:
    print("Model not recognised.")
    quit()

pytrain.train_distribution( sentence_net, s, 25000, collate_fn=collate, epoch_end_fn=epoch_end_callback)
