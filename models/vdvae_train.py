import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import vdvae.train as vdvae_train
import vdvae.hps as hps
import vdvae.vae as vdvae
import pygenbrix_layer as pygl
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp


class VDVAEModel(nn.Module):
    def __init__(self, vae, ema_vae):
        super(VDVAEModel, self).__init__()
        self.vae, self.ema_vae = vae, ema_vae


class VDVAETrainer(pl.LightningModule):
    def __init__(self, model, dataset, batch_size=8, learning_rate=0.0002):
        super(VDVAETrainer, self).__init__()
        self.model = model
        self.dataset = dataset
        self.train_set, self.val_set = self.get_datasets()
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def get_datasets(self):
        dataset_size = len(self.dataset)
        training_size = np.round(dataset_size*0.9).astype(int)
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [training_size, dataset_size-training_size],
            generator=torch.Generator().manual_seed(42) ) 
        return (train_set, val_set)

    def training_step(self, batch, batch_indx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        stats = vdvae_train.training_step(h, (x-.5)*4, x, self.model.vae, self.model.ema_vae, self.optimizers(), -1)
        self.log('log_prob', -stats["elbo"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kl', stats["rate"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('recon_error', stats["distortion"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_indx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        stats = vdvae_train.eval_step((x-.5)*4, x, self.model.ema_vae)
        self.log("validation_log_prob", -stats["elbo"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_kl", stats["rate"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_recon_error", stats["distortion"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size = self.batch_size, num_workers=4, drop_last=True, pin_memory=True)


class LogSamplesVAECallback(pl.Callback):
    def __init__(self, step_freq):
        super(LogSamplesVAECallback, self).__init__()
        self.step_freq = step_freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.step_freq == 0) and (batch_idx % trainer.accumulate_grad_batches == 0):
            samples = pl_module.model.ema_vae.forward_uncond_samples(8, t=1.0)
            samples_grid = torchvision.utils.make_grid(samples, padding=10, nrow=4)
            pl_module.logger.experiment.add_image("train_sample", samples_grid, pl_module.global_step, dataformats="CHW")


ap = argparse.ArgumentParser(description="VDVAE")
ap.add_argument("--tensorboard_log")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--lr", default=.0002, type=float)
ap.add_argument("--ema_rate", default=.9999, type=float)
ap.add_argument("--fast_dev_run", action="store_true")
ap.add_argument("--dataset")
ap.add_argument("--rv_distribution")
ns = ap.parse_args()
h = hps.Hyperparams()
h.width = 384
h.zdim = 16
h.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
h.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
h.image_size = 32
h.ema_rate = ns.ema_rate
h.custom_width_str = ""
h.no_bias_above = 64
h.bottleneck_multiple = 0.25
h.num_mixtures = 10
h.grad_clip = 200.0
h.skip_threshold = 400.0

if ns.dataset == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root='/home/julian/ImageDataSets/CIFAR10', train=True,
        download=False, transform=torchvision.transforms.ToTensor())
    h.image_channels = 3
elif ns.dataset == "celeba32":
    dataset = torchvision.datasets.ImageFolder(root="/home/julian/ImageDataSets/celeba/",
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad((-15, -40,-15-1, -30-1)),
            torchvision.transforms.Resize(32), torchvision.transforms.ToTensor(),
        ]))
    h.image_channels = 3
elif ns.dataset == "mnist32":
    dataset = torchvision.datasets.MNIST('/home/julian/ImageDataSets/MNIST',
    train=True, download=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 1.0-((x<0.5)*1.0))]))
    h.image_channels = 1
    h.image_size = 32
else:
    print("Dataset not recognized.")
    exit(1)

if ns.rv_distribution == "bernoulli":
    rv_distribution = dl.IndependentBernoulliLayer()
elif ns.rv_distribution == "q3":
    rv_distribution = dl.IndependentQuantizedLayer(num_buckets = 8)
elif ns.rv_distribution == "spiq3":
    rv_distribution = sp.SpatialIndependentDistributionLayer([h.image_channels, h.image_width, image_width], dl.IndependentQuantizedLayer(num_buckets = 8), num_params=30)
elif ns.rv_distribution == "PixelCNNDiscMixDistribution":
    rv_distribution = pixel_cnn.PixelCNNDiscreteMixLayer()
elif ns.rv_distribution == "VDVAEDiscMixDistribution":
    rv_distribution = pygl.VDVAEDiscMixtureLayer(h)
else:
    print("rv distribution not recognized")
    quit()

vae = vdvae.VAE(h, rv_distribution)
ema_vae = vdvae.VAE(h, rv_distribution)
ema_vae.requires_grad = False
model = VDVAEModel(vae, ema_vae)
trainer = VDVAETrainer(model, dataset, batch_size=8, learning_rate=ns.lr)
pl.Trainer(fast_dev_run = ns.fast_dev_run, gpus=1, accumulate_grad_batches = 1, max_epochs=ns.max_epochs, default_root_dir=ns.tensorboard_log, 
    callbacks=[LogSamplesVAECallback(1000)]).fit(trainer)
