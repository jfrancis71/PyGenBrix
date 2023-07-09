# Generalization demonstration, inspired by book Parallel Distributed Processing, Volume 1: Foundations
# by Rumelhart, McClelland & PDP Research Group, MIT 1986, page 340, Symmetry example.

import argparse
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer import Predictive
from pyro.nn import PyroSample, PyroModule


dataset_x = torch.tensor([[float(d) for d in list(np.binary_repr(n, 12))] for n in range(4096)])
dataset_y = torch.prod((dataset_x == torch.flip(dataset_x, dims=(1,)))*1.0, dim=1)

subset_mask = torch.distributions.bernoulli.Bernoulli(probs=torch.ones([4096])*0.9).sample().type(torch.bool)

train_dataset_x = dataset_x[subset_mask]
train_dataset_y = dataset_y[subset_mask]
validation_dataset_x = dataset_x[torch.logical_not(subset_mask)]
validation_dataset_y = dataset_y[torch.logical_not(subset_mask)]


class BNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.layer1 = PyroModule[nn.Linear](12, 24)
        self.layer2 = PyroModule[nn.Linear](24, 1)
        self.layer1.bias = PyroSample(dist.Normal(0., 10.).expand([24]).to_event(1))
        self.layer1.weight = PyroSample(dist.Normal(0., 10.).expand([24, 12]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., 10.).expand([1, 24]).to_event(2))

    def forward(self, x, y=None):
        x = nn.Sigmoid()(self.layer1(x))
        logit = self.layer2(x).squeeze()
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
        return logit


def mean_field_guide(*_):
    layer1_bias = pyro.param('param_layer_1_bias', torch.zeros([24]))
    layer1_weight = pyro.param('param_layer1_weight', torch.zeros([24, 12]))
    layer2_bias = pyro.param('param_layer2_bias', torch.zeros([1]))
    layer2_weight = pyro.param('param_layer2_weight', torch.zeros([1, 24]))
    pyro.sample("layer1.bias", dist.Normal(layer1_bias, torch.ones([24])).to_event(1))
    pyro.sample("layer1.weight", dist.Normal(layer1_weight, torch.ones([24, 12])).to_event(2))
    pyro.sample("layer2.bias", dist.Normal(layer2_bias, torch.ones([1])).to_event(1))
    pyro.sample("layer2.weight", dist.Normal(layer2_weight, torch.ones([1, 24])).to_event(2))


ap = argparse.ArgumentParser(description="Generalization Demonstration")
ap.add_argument("--train")
ns = ap.parse_args()

print("Training set sum=", train_dataset_y.sum(), "Validation set sum=", validation_dataset_y.sum())


def svi_train():
    model = BNN()
    svi = SVI(model,
              mean_field_guide,
              optim.Adam({"lr": .01}),
              loss=Trace_ELBO())
    pyro.clear_param_store()
    num_iters = 500000
    for i in range(num_iters):
        elbo = svi.step(train_dataset_x, train_dataset_y)
        if i % 5000 == 0:
            train_ans = Predictive(model, guide=mean_field_guide, num_samples=1)(train_dataset_x, None)["obs"][0]
            valid_ans = Predictive(model, guide=mean_field_guide, num_samples=1)(validation_dataset_x, None)["obs"][0]
            print("iter=", i,
                  "elbo=", elbo,
                  "train_ans=", torch.abs(train_ans-train_dataset_y).sum(),
                  "valid_ans=", torch.abs(valid_ans-validation_dataset_y).sum())


def mcmc_train():
    model = BNN()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=50)
    mcmc.run(train_dataset_x, train_dataset_y)
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    print("train_ans=", (predictive(train_dataset_x)["obs"][0] - train_dataset_y).abs().sum(),
          "valid_ans=", (predictive(validation_dataset_x)["obs"][0]-validation_dataset_y).abs().sum())


if ns.train == "svi":
    svi_train()
elif ns.train == "mcmc":
    mcmc_train()
else:
    print("train not recognised.")
