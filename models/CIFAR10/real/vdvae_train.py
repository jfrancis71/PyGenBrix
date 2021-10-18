import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
#import vdvae.train_helpers as th
from vdvae import train_helpers as th
import vdvae.data as vdvae_data
import vdvae.train as vdvae_train


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
        stats = vdvae_train.training_step(H, (x-.5)*4, (x-.5)*2, self.model.vae, self.model.ema_vae, self.optimizers(), -1)
        self.log('elbo', stats["elbo"])
        self.log('kl', stats["rate"])
        self.log('recon_error', stats["distortion"])

    def validation_step(self, batch, batch_indx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        stats = vdvae_train.eval_step((x-.5)*4, (x-.5)*2, self.model.ema_vae)
        self.log('elbo', stats["elbo"])
        self.log('kl', stats["rate"])
        self.log('recon_error', stats["distortion"])


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
            samples = torch.tensor(samples).permute((0,3,1,2))
            samples_grid = torchvision.utils.make_grid(samples, padding=10, nrow=4)
            pl_module.logger.experiment.add_image("train_sample", samples_grid, pl_module.global_step, dataformats="CHW")


dataset = torchvision.datasets.CIFAR10(root='/home/julian/ImageDataSets/CIFAR10', train=True,
                                        download=False, transform=torchvision.transforms.ToTensor())
H, logprint = th.set_up_hyperparams()
H.image_channels=3
H.image_size=32
vae, ema_vae = th.load_vaes(H, logprint)
model = VDVAEModel(vae, ema_vae)
trainer = VDVAETrainer(model, dataset)
pl.Trainer(fast_dev_run = False, gpus=1, accumulate_grad_batches = 1, max_epochs=10,
    callbacks=[LogSamplesVAECallback(1000)]).fit(trainer)
