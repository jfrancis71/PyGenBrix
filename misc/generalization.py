# Generalization demonstration, inspired by book Parallel Distributed Processing, Volume 1: Foundations
# by Rumelhart, McClelland & PDP Research Group, MIT 1986, page 340, Symmetry example.

import argparse
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer import Predictive


dataset_x = torch.tensor([[float(d) for d in list(np.binary_repr(n, 12))] for n in range(4096)])
dataset_y = torch.prod((dataset_x == torch.flip(dataset_x, dims=(1,)))*1.0, dim=1)

subset_mask = torch.distributions.bernoulli.Bernoulli(probs=torch.ones([4096])*0.9).sample().type(torch.bool)

train_dataset_x = dataset_x[subset_mask]
train_dataset_y = dataset_y[subset_mask]
validation_dataset_x = dataset_x[torch.logical_not(subset_mask)]
validation_dataset_y = dataset_y[torch.logical_not(subset_mask)]


def model(x, y=None):
    weights1 = pyro.sample("weights1", dist.Normal(torch.zeros([24, 12]), 30.0 * torch.ones([24, 12])).to_event(2))
    biases1 = pyro.sample("biases1", dist.Normal(torch.zeros([24]), 30.0 * torch.ones([24])).to_event(1))
    weights2 = pyro.sample("weights2", dist.Normal(torch.zeros([24]), 30.0 * torch.ones([24])).to_event(1))
    biases2 = pyro.sample("biases2", dist.Normal(torch.zeros([1]), 30.0 * torch.ones([1])).to_event(1))
    out1 = torch.matmul(weights1, x.permute(1, 0)).permute(1, 0) + biases1
    out1 = torch.sigmoid(out1)
    out = torch.matmul(out1, weights2) + biases2[0]
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=out), obs=y)


def guide(x, y):
    biases1 = pyro.param('param_biases1', torch.zeros([24]))
    weights1 = pyro.param('param_weights1', torch.zeros([24, 12]))
    biases2 = pyro.param('param_biases2', torch.zeros([1]))
    weights2 = pyro.param('param_weights2', torch.zeros([24]))
    pyro.sample("weights1", dist.Normal(weights1, torch.ones([24, 12])).to_event(2))
    pyro.sample("biases1", dist.Normal(biases1, torch.ones([24])).to_event(1))
    pyro.sample("biases2", dist.Normal(biases2, torch.ones([1])).to_event(1))
    pyro.sample("weights2", dist.Normal(weights2, torch.ones([24])).to_event(1))


ap = argparse.ArgumentParser(description="Text Generator")
ap.add_argument("--train")
ns = ap.parse_args()

print("Training set sum=", train_dataset_y.sum(), "Validation set sum=", validation_dataset_y.sum())


def svi_train():
    svi = SVI(model,
              guide,
              optim.Adam({"lr": .01}),
              loss=Trace_ELBO())
    pyro.clear_param_store()
    num_iters = 500000
    for i in range(num_iters):
        elbo = svi.step(train_dataset_x, train_dataset_y)
        if i % 5000 == 0:
            train_ans = Predictive(model, guide=guide, num_samples=1)(train_dataset_x, None)["obs"][0]
            valid_ans = Predictive(model, guide=guide, num_samples=1)(validation_dataset_x, None)["obs"][0]
            print("iter=", i,
                  "elbo=", elbo,
                  "train_ans=", torch.abs(train_ans-train_dataset_y).sum(),
                  "valid_ans=", torch.abs(valid_ans-validation_dataset_y).sum())


def mcmc_train():
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
