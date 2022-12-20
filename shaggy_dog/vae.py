# Some attempts at discrete VAE.

import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

dataset = torchvision.datasets.MNIST('./data', 
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0))]),
    download=True)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True)

# Attempt at summing over z
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_net = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(10,784),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def loss(self, x, target):
        z_categorical = torch.distributions.categorical.Categorical(logits=self.encoder_net(x))
#        encoder_loss = -z_categorical.log_prob(target).mean()
#        z_sample = z_categorical.sample()
        decoder_log_prob = torch.zeros(target.shape[0])
        for z in range(10):
            z_sample = torch.tensor(z).repeat(target.shape[0])
            one_hot_targets = torch.nn.functional.one_hot(z_sample, num_classes=10)
            image_logits = torch.reshape(self.decoder(one_hot_targets.type(torch.float32)), (target.shape[0], 1,28,28))
            image_dist = torch.distributions.independent.Independent(torch.distributions.bernoulli.Bernoulli(logits=image_logits), reinterpreted_batch_ndims=3)
            decoder_log_prob += torch.exp(z_categorical.log_prob(z_sample))*image_dist.log_prob(x)
        kl_div = torch.distributions.kl_divergence(z_categorical, torch.distributions.categorical.Categorical(logits=torch.zeros(x.shape[0], 10)))
        elbo = decoder_log_prob - kl_div
        return -elbo.mean()

vae = VAE()

def train():
    total_loss = 0.0
    total_iter = 0
    for (image, target) in iter(dataloader):
        vae.optimizer.zero_grad()
        loss = vae.loss(image, target)
        loss.backward()
        vae.optimizer.step()
        total_loss += loss.item()
        total_iter += 1
    print(total_loss/total_iter)
