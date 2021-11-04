import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import vdvae.train as vdvae_train
import vdvae.hps as hps
import PyGenBrix.dist_layers.vdvae as vdvae
import pygenbrix_layer as pygl
import PyGenBrix.dist_layers.common_layers as dl
import PyGenBrix.dist_layers.spatial_independent as sp
import PyGenBrix.models.parser as parser


class EMAModel(nn.Module):
    def __init__(self, model, ema_model):
        super(EMAModel, self).__init__()
        self.model, self.ema_model = model, ema_model


class EMATrainer(pl.LightningModule):
    def __init__(self, model, dataset, batch_size=8, learning_rate=0.0002, ema_rate=0.99):
        super(EMATrainer, self).__init__()
        self.model = model
        self.dataset = dataset
        self.train_set, self.val_set = self.get_datasets()
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip = 200
        self.skip_threshold = 400
        self.ema_rate = ema_rate

    def get_datasets(self):
        dataset_size = len(self.dataset)
        training_size = np.round(dataset_size*0.9).astype(int)
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [training_size, dataset_size-training_size],
            generator=torch.Generator().manual_seed(42) ) 
        return (train_set, val_set)

    def training_step(self, batch, batch_indx):
        x, y = batch
        ndims = np.prod(x.shape[1:])
        self.model.model.zero_grad()
        log_prob = self.model.model.log_prob(x)["log_prob"].mean()
        log_prob_per_pixel = log_prob/ndims
        loss = -log_prob_per_pixel
        loss.backward()       
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.grad_clip).item()
        log_prob_per_pixel_nans = torch.isnan(log_prob_per_pixel)
        self.log("log_prob_nans", log_prob_per_pixel_nans, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        skipped_update = 1
        if (log_prob_per_pixel_nans == False and grad_norm < self.skip_threshold):
            self.optimizers().step()
            skipped_update = 0
            vdvae_train.update_ema(self.model.model, self.model.ema_model, self.ema_rate)
        self.log('skipped_update', skipped_update, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('log_prob', log_prob_per_pixel, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_indx):
        x, y = batch
        log_prob = self.model.ema_model.log_prob(x)
        self.log("validation_log_prob", log_prob["log_prob"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
            samples = pl_module.model.ema_model.sample([8], temperature=1.0)
            samples_grid = torchvision.utils.make_grid(samples, padding=10, nrow=4)
            pl_module.logger.experiment.add_image("train_sample", samples_grid, pl_module.global_step, dataformats="CHW")


ap = argparse.ArgumentParser(description="EMATrainer")
ap.add_argument("--tensorboard_log")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--lr", default=.0002, type=float)
ap.add_argument("--ema_rate", default=.9999, type=float)
ap.add_argument("--fast_dev_run", action="store_true")
ap.add_argument("--model")
ap.add_argument("--dataset")
ap.add_argument("--rv_distribution")
ns = ap.parse_args()

event_shape, dataset = parser.get_dataset(ns)
rv_distribution = parser.get_rv_distribution(ns, event_shape)
if ns.model != "vdvae":
    print("This program only supports vdvae")
    quit()
vae = vdvae.VDVAE(event_shape, rv_distribution)
ema_vae = vdvae.VDVAE(event_shape, rv_distribution)
ema_vae.requires_grad = False
model = EMAModel(vae, ema_vae)
trainer = EMATrainer(model, dataset, batch_size=8, learning_rate=ns.lr, ema_rate=ns.ema_rate)
pl.Trainer(fast_dev_run = ns.fast_dev_run, gpus=1, accumulate_grad_batches = 1, max_epochs=ns.max_epochs, default_root_dir=ns.tensorboard_log, 
    callbacks=[LogSamplesVAECallback(1000)]).fit(trainer)
