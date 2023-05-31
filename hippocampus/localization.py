import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as Categorical
import PyGenBrix.hippocampus.scene_recognition as scene_recognition
import PyGenBrix.dist_layers.sequence_hmm as hmm_model


class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, seq_images_dataset):
        self.num_seq = int(len(seq_images_dataset) / 50)
        self.train_tensor = torch.stack([seq_images_dataset[t][0] for t in range(200)])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.train_tensor[:200]


# Ideally you would have cache'ing and non cacheing versions,
# this implementation has batch size 1, ideally generalize.
class FeatureTransformDataset(torch.utils.data.Dataset):
    def __init__(self, stream_images_dataset, scene_recognition):
        self.features = scene_recognition(stream_images_dataset[0][:,0]).detach()
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.features[:200]


class MatrixObservationModel(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.observation_matrix = nn.Parameter(torch.rand([num_states,16,13,12]))

    def emission_logits(self, observation):  # (Batch, observation)
        return torch.stack([(observation[b]-self.observation_matrix)**2 for b in range(observation.shape[0])]).sum([2,3,4])/-500


class StateModel(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.prior_state_vector = torch.zeros([num_states])-10.0
        self.prior_state_vector[0] = 10.0
        self.state_transition_matrix = 0.1*torch.eye(num_states) + .8*torch.roll(torch.eye(num_states), 1,1) + .08*torch.roll(torch.eye(num_states), 2,1) + (.02/num_states)*torch.ones([num_states, num_states])

    def prior_states(self):
        return self.prior_state_vector

    def state_transitions(self):
        return Categorical(probs=self.state_transition_matrix).logits


class Localization(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_recognizer = scene_recognition.SceneRecognition()
        self.hmm = hmm_model.HMM(StateModel(35), MatrixObservationModel(35))
        self.current_state_distribution = torch.exp(self.hmm.current_state_distribution)
        
    def observe(self, image_tensor):
        sr = self.scene_recognizer(image_tensor[0].unsqueeze(0))[0]
        self.hmm.observe(sr)
        self.current_state_distribution = torch.exp(self.hmm.current_state_distribution)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
