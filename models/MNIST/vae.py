import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl
import PyGenBrix.Train as Train
import PyGenBrix.models.residual_block as rb


ap = argparse.ArgumentParser(description="VAE")
ap.add_argument("--tensorboard_log")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--z_samples", default=1, type=int)
ap.add_argument("--latents", default=64, type=int)
ap.add_argument("--lr", default=.0002, type=float)
ns = ap.parse_args()

mnist_dataset = torchvision.datasets.MNIST('/home/julian/ImageDataSets/MNIST', train=True, download=False,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0))
                   ]))


class StochasticBinaryLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bin_latent_code = torch.distributions.bernoulli.Bernoulli(logits=x).sample()
        probs = F.sigmoid(x)
        stg_latent_code = bin_latent_code + probs - probs.detach()
        return stg_latent_code


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.l0 = nn.Linear(256*4*4, ns.latents)
        self.b1 = rb.ResidualBlock(64)
        self.b2 = rb.ResidualBlock(128)

    def forward(self, x):
        x = self.c1(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.c2(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.b1(x)
        x = self.c3(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.b2(x)
        x = self.c4(x)
        x = nn.LeakyReLU(0.02)(x)
        x = x.view(x.size(0), -1)
        x = self.l0(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = 128
        self.fc = nn.Linear(in_features=64, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x


class VAE(nn.Module):
    def __init__(self, z_samples=1):
        super(VAE, self).__init__()
        self.bin = StochasticBinaryLayer()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.z_samples = z_samples

    def log_prob1(self, x):
        logits = self.encoder(x)
        zsample = self.bin(logits)
        decode = self.decoder(zsample)
        BCE = torch.sum(F.binary_cross_entropy(decode, x, reduction='none'), axis=[1,2,3])
        KLD = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.bernoulli.Bernoulli(logits=logits),
            torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([64,ns.latents]).to(x.device))), axis=1)
        log_prob = -BCE - KLD
        return {"log_prob": log_prob,
                "recon_log_prob": -BCE,
                "KLD": KLD}

    def log_prob(self, x):
        sample_log_probs = [ self.log_prob1(x)["log_prob"] for i in range(self.z_samples) ]
        log_prob = torch.logsumexp(torch.stack(sample_log_probs, dim=0), dim=0) - torch.log(torch.tensor(self.z_samples))
        return {"log_prob": log_prob}

    def sample(self, temperature=1.0):
        z = torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([1,ns.latents])).sample().to("cuda")
        sample = self.decoder(z)
        sample = torch.distributions.bernoulli.Bernoulli(probs=sample).sample().view(-1,1,28,28)
        return sample

    def sample_reconstruction(self, x):
        logits = self.encoder(x)
        z = self.bin(logits)
        sample = self.decoder(z)
        sample = torch.distributions.bernoulli.Bernoulli(probs=sample).sample().view(-1,1,28,28)
        return sample


mymodel = VAE(ns.z_samples)

trainer = Train.LightningDistributionTrainer( mymodel, mnist_dataset, learning_rate = ns.lr, batch_size = 64 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches = 1, max_epochs=ns.max_epochs, 
          callbacks=[
                     Train.LogSamplesEpochCallback(temperature=1.0),
                     Train.LogReconstructionEpochCallback(),
          ], default_root_dir=ns.tensorboard_log).fit( trainer )
