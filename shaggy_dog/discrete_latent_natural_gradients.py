# Attempt at discrete latent mnist model using natural gradients to learn.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

transform=transforms.Compose([
        transforms.Resize(8),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0))
        ])

dataset1 = datasets.MNIST("\home\julian\data", train=True, download=True,
                       transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=2048)

class Fred(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.rand([10,8,8]))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        
    def log_prob(self, x):
        ind = torch.distributions.independent.Independent(torch.distributions.bernoulli.Bernoulli(logits=self.params),reinterpreted_batch_ndims=2)
        digit_dist = torch.distributions.mixture_same_family.MixtureSameFamily(
            torch.distributions.categorical.Categorical(torch.ones([10])/10.0), ind)
        return digit_dist.log_prob(x)

model = Fred()

def train_epoch():
    total_loss = 0.0
    iters = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        iters += 1
        model.optimizer.zero_grad()
        log_prob = model.log_prob(data[:,0])
        loss = -log_prob.mean()
        loss.backward()
        total_loss += loss.item()
        v = model.params.grad.flatten().detach()
        cov = torch.outer(v,v)
#        global savev
#        savev = v
        f = torch.inverse(cov)
        model.params.grad = torch.matmul(f,v).reshape(10,8,8)
#        print(model.params.grad[0,0,0])
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        model.optimizer.step()
    return total_loss/iters

for i in range(500):
    print(train_epoch())
