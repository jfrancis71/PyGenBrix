import argparse
import copy
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
import PyGenBrix.Train as Train


class EMATrainer(Train.LightningTrainer):
    def __init__(self, model, train_model, dataset, batch_size=8, learning_rate=0.0002, ema_rate=0.99):
        super(EMATrainer, self).__init__(model, dataset, False, learning_rate, batch_size)
        self.train_model = train_model
        self.model.requires_grad = False
        self.automatic_optimization = False
        self.grad_clip = 200
        self.skip_threshold = 400
        self.ema_rate = ema_rate

    def training_step(self, batch, batch_indx):
        x, y = batch
        ndims = np.prod(x.shape[1:])
        self.train_model.zero_grad()
        log_prob = self.train_model.log_prob(x)["log_prob"].mean()
        log_prob_per_pixel = log_prob/ndims
        loss = -log_prob_per_pixel
        loss.backward()       
        grad_norm = torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), self.grad_clip).item()
        log_prob_per_pixel_nans = torch.isnan(log_prob_per_pixel)
        self.log("log_prob_nans", log_prob_per_pixel_nans, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        skipped_update = 1
        if (log_prob_per_pixel_nans == False and grad_norm < self.skip_threshold):
            self.optimizers().step()
            skipped_update = 0
            vdvae_train.update_ema(self.train_model, self.model, self.ema_rate)
        self.log('skipped_update', skipped_update, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('log_prob', log_prob_per_pixel, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_indx):
        x, y = batch
        log_prob = self.model.log_prob(x)
        self.log("validation_log_prob", log_prob["log_prob"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.train_model.parameters(), lr = self.learning_rate)


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
model = vdvae.VDVAE(event_shape, rv_distribution)
train_model = vdvae.VDVAE(event_shape, rv_distribution)
trainer = EMATrainer(model, train_model, dataset, batch_size=8, learning_rate=ns.lr, ema_rate=ns.ema_rate)
pl.Trainer(fast_dev_run = ns.fast_dev_run, gpus=1, accumulate_grad_batches = 1, max_epochs=ns.max_epochs, default_root_dir=ns.tensorboard_log, 
    callbacks=[Train.LogSamplesTrainingCallback(every_global_step=100, temperature=1.0)]).fit(trainer)
