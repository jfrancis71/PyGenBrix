import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from pyro.nn import PyroSample, PyroModule
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer import Predictive


class BNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.layer1 = PyroModule[nn.Linear](28*28, 1)
        self.layer1.bias = PyroSample(dist.Normal(0., 1.).expand([10]).to_event(1))
        self.layer1.weight = PyroSample(dist.Normal(0., 1.).expand([10, 28*28]).to_event(2))

    def forward(self, x, y=None):
        x = x[:, 0].flatten(1)
        logit = self.layer1(x)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logit), obs=y)
        return logit


def svi_train(model):
    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    svi = SVI(model,
              auto_guide,
              optim.Adam({"lr": .01}),
              loss=Trace_ELBO())
    pyro.clear_param_store()
    num_iters = 500000
    for i in range(num_iters):
        elbo = svi.step(train_dataset_x[:5000], train_dataset_y[:5000])
        if i % 50 == 0:
            train_ans = Predictive(model, guide=auto_guide, num_samples=1)(train_dataset_x[5000:6000], None)["obs"][0]
            print("iter=", i,
                  "elbo=", elbo,
                  "train_ans=", (train_ans == train_dataset_y[5000:6000]).sum())


def mcmc_train(model):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=50)
    mcmc.run(train_dataset_x[:5000], train_dataset_y[:5000])
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    train_ans = predictive(train_dataset_x[5000:6000])["obs"][0]
    print("train_ans=", (train_ans == train_dataset_y[5000:6000]).sum())


dataset = datasets.MNIST('/Users/julian/ImageDataSets/MNIST',
                         train=True, download=False,
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: 1.0-((x < 0.5)*1.0))]))
train_dataset_x = torch.stack([dataset[idx][0] for idx in range(len(dataset))])
train_dataset_y = torch.stack([torch.tensor(dataset[idx][1]) for idx in range(len(dataset))])
bnn_model = BNN()

ap = argparse.ArgumentParser(description="Generalization Demonstration")
ap.add_argument("--train")
ns = ap.parse_args()

if ns.train == "svi":
    svi_train(bnn_model)
elif ns.train == "mcmc":
    mcmc_train(bnn_model)
else:
    print("train not recognised.")
