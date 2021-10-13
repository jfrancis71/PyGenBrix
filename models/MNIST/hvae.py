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


class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3, self).__init__()
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.l0 = nn.Conv2d(256, ns.latents, kernel_size=4, stride=1, padding=0)
        self.b2 = rb.ResidualBlock(128)

    def forward(self, x):
        x = self.b2(x)
        x = self.c4(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.l0(x)
        return x


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b1 = rb.ResidualBlock(64)

    def forward(self, x):
        x = self.c2(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.b1(x)
        x = self.c3(x)
        x = nn.LeakyReLU(0.02)(x)
        return x


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.c1(x)
        x = nn.LeakyReLU(0.02)(x)
        return x


class EncoderSampleZ2(nn.Module):
    def __init__(self):
        super(EncoderSampleZ2, self).__init__()
        self.c1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.p = nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.map = Decoder3()

    def forward(self, z3, x2):
        z1 = self.map(z3) + self.p(x2)
        z1 = self.c1(z1)
        z1 = nn.LeakyReLU(0.02)(z1)
        z1 = self.c2(z1)
        z1 = nn.LeakyReLU(0.02)(z1)
        z1 = self.c3(z1)
        return z1


class EncoderSampleZ1(nn.Module):
    def __init__(self):
        super(EncoderSampleZ1, self).__init__()
        self.c1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.p = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.map = Decoder2()

    def forward(self, z2, x1):
        z1 = self.map(z2) + self.p(x1)
        z1 = self.c1(z1)
        z1 = nn.LeakyReLU(0.02)(z1)
        z1 = self.c2(z1)
        z1 = nn.LeakyReLU(0.02)(z1)
        z1 = self.c3(z1)
        return z1


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.conv1 = nn.Conv2d(64, 128*2*7*7, kernel_size=1)
        self.conv2 = nn.Conv2d(128*2, 16, kernel_size=3, padding=1)
            
    def forward(self, z3):
        x = nn.LeakyReLU(0.02)(self.conv1(z3).view(-1,128*2,7,7))
        x = self.conv2(x)
        return x


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.b1 = rb.ResidualBlock(64)

    def forward(self, z2):
        x = F.relu(self.conv2(z2))
        x = self.b1(x)
        x = self.conv1(x)
        return x

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.c1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.c1(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.c2(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.c3(x)
        return x


class VAE(nn.Module):
    def __init__(self, z_samples=1):
        super(VAE, self).__init__()
        self.bin = StochasticBinaryLayer()
        self.encoder3 = Encoder3()
        self.encoder2 = Encoder2()
        self.encoder1 = Encoder1()
        self.encoder_samplez1 = EncoderSampleZ1()
        self.encoder_samplez2 = EncoderSampleZ2()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.z_samples = z_samples

    def log_prob1(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        qlogits3 = self.encoder3(x2)
        zsample3 = self.bin(qlogits3)
        qlogits2 = self.encoder_samplez2(zsample3, x2)
        zsample2 = self.bin(qlogits2)
        qlogits1 = self.encoder_samplez1(zsample2, x1)
        zsample1 = self.bin(qlogits1)
        plogits2 = self.decoder3(zsample3)
        plogits1 = self.decoder2(zsample2)
        xlogits = self.decoder1(zsample1)
        BCE = torch.sum(F.binary_cross_entropy(torch.sigmoid(xlogits), x, reduction='none'), axis=[1,2,3])
        KLD3 = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.bernoulli.Bernoulli(logits=qlogits3),
            torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([64,ns.latents,1,1]).to(x.device))), axis=[1, 2, 3])
        KLD2 = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.bernoulli.Bernoulli(logits=qlogits2),
            torch.distributions.bernoulli.Bernoulli(logits=plogits2)), axis=[1,2,3])
        KLD1 = torch.sum(torch.distributions.kl_divergence(
            torch.distributions.bernoulli.Bernoulli(logits=qlogits1),
            torch.distributions.bernoulli.Bernoulli(logits=plogits1)), axis=[1,2,3])

        log_prob = -BCE - KLD2 - KLD1 - KLD3
        return {"log_prob": log_prob,
                "recon_log_prob": -BCE,
                "KLD": KLD2}

    def log_prob(self, x):
        sample_log_probs = [ self.log_prob1(x)["log_prob"] for i in range(self.z_samples) ]
        log_prob = torch.logsumexp(torch.stack(sample_log_probs, dim=0), dim=0) - torch.log(torch.tensor(self.z_samples))
        return {"log_prob": log_prob}

    def sample(self, temperature=1.0):
        z3 = torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([1, ns.latents, 1, 1])).sample().to("cuda")
        sample = self.bin(self.decoder3(z3))
        sample = self.bin(self.decoder2(sample))
        sample = self.bin(self.decoder1(sample))
        sample = torch.distributions.bernoulli.Bernoulli(probs=sample).sample().view(-1,1,28,28)
        return sample

    def sample_reconstruction(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        logits3 = self.encoder3(x2)
        z3 = self.bin(logits3)
        sample = self.bin(self.decoder3(z3))
        sample = self.bin(self.decoder2(sample))
        sample = self.bin(self.decoder1(sample))
        sample = torch.distributions.bernoulli.Bernoulli(probs=sample).sample().view(-1,1,28,28)
        return sample


mymodel = VAE(ns.z_samples)

trainer = Train.LightningDistributionTrainer( mymodel, mnist_dataset, learning_rate = ns.lr, batch_size = 64 )

pl.Trainer( fast_dev_run = False, gpus=1, accumulate_grad_batches = 1, max_epochs=ns.max_epochs, 
          callbacks=[
                     Train.LogSamplesEpochCallback(temperature=1.0),
                     Train.LogReconstructionEpochCallback(),
          ], default_root_dir=ns.tensorboard_log).fit( trainer )
