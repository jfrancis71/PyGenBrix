import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as Categorical
import PyGenBrix.hippocampus.scene_recognition as scene_recognition


class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, seq_images_dataset):
        self.num_seq = int(len(seq_images_dataset) / 50)
        self.train_tensor = torch.stack([seq_images_dataset[t][0] for t in range(500)])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.train_tensor[:500]


# Ideally you would have cache'ing and non cacheing versions,
# this implementation has batch size 1, ideally generalize.
class FeatureTransformDataset(torch.utils.data.Dataset):
    def __init__(self, stream_images_dataset, scene_recognition):
        self.features = scene_recognition(stream_images_dataset[0][:,0]).detach()
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.features[:100]


class HMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior_state_distribution = torch.zeros([35])-10.0
        self.prior_state_distribution[0] = 10.0
        self.transition_matrix = 0.1*torch.eye(35) + .8*torch.roll(torch.eye(35), 1,1) + .08*torch.roll(torch.eye(35), 2,1) + (.02/35)*torch.ones([35,35])
        self.observation_matrix = nn.Parameter(torch.rand([35,16,13,12]))
        
    def test(self, observations):
        diff1 = torch.sum((observations[0,0] - self.observation_matrix[0].detach())**2)
        diff2 = torch.sum((observations[0,15] - self.observation_matrix[15].detach())**2)
        diff3 = torch.sum((observations[0,0] - self.observation_matrix[15].detach())**2)
        print("diff=", diff1, diff2, diff3)

    def log_prob(self, observations):
        alpha = torch.stack([(observations[b,0]-self.observation_matrix)**2 for b in range(observations.shape[0])]).sum([2,3,4])/-500 + self.prior_state_distribution
        for observation_idx in range(1, observations.shape[1]):
            alpha = torch.stack([(observations[b, observation_idx]-self.observation_matrix)**2 for b in range(observations.shape[0])]).sum([2,3,4])/-500 + \
                torch.logsumexp(torch.transpose(Categorical(probs=self.transition_matrix).logits, 0, 1) + alpha.unsqueeze(1), dim=2)
        return torch.logsumexp(alpha, dim=1)


class Localization(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_recognizer = scene_recognition.SceneRecognition()
        self.hmm = HMM()
        self.state_distribution = torch.ones([35])/35
        
    def observe(self, image_tensor):
        sr = self.scene_recognizer(image_tensor[0].unsqueeze(0))[0]
        observation_values = torch.stack([ ((sr-self.hmm.observation_matrix[state])**2).sum() for state in range(35) ]).detach()
        observation_vector = np.exp(-observation_values/500.)
        unnormalized = torch.matmul(self.state_distribution, self.hmm.transition_matrix)*observation_vector
        self.state_distribution = unnormalized/torch.sum(unnormalized)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
