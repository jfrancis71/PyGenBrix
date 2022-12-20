import numpy as np
import torch
import torch.nn as nn
import vdvae.train as vdvae_train
import PyGenBrix.utils.trainer as Train


class EMATrainer(Train.LightningDistributionTrainer):
    def __init__(self, model, train_model, dataset, batch_size=8, learning_rate=0.0002, ema_rate=0.99, accumulate_grad_batches=1):
        super(EMATrainer, self).__init__(model, dataset, False, learning_rate, batch_size)
        self.train_model = train_model
        self.model.requires_grad = False
        self.automatic_optimization = False
        self.grad_clip = 200
        self.skip_threshold = 400
        self.ema_rate = ema_rate
        self.accumulate_grad_batches = accumulate_grad_batches

    def training_step(self, batch, batch_indx):
        x, y = batch
        ndims = np.prod(x.shape[1:])
        log_prob = self.train_model.log_prob(x)["log_prob"].mean()
        log_prob_per_pixel = log_prob/ndims
        loss = -log_prob_per_pixel/self.accumulate_grad_batches
        loss.backward()       
        grad_norm = torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), self.grad_clip).item()
        log_prob_per_pixel_nans = torch.isnan(log_prob_per_pixel)
        self.log("log_prob_nans", log_prob_per_pixel_nans, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        skipped_update = 0
        if (log_prob_per_pixel_nans == False and grad_norm < self.skip_threshold):
            if (batch_indx + 1) % self.accumulate_grad_batches == 0:
                self.optimizers().step()
                self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                self.train_model.zero_grad()
                skipped_update = 0
                vdvae_train.update_ema(self.train_model, self.model, self.ema_rate)
        else:
            self.train_model.zero_grad()
            skipped_update = 1
        self.log('skipped_update', skipped_update, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('log_prob', log_prob_per_pixel, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.train_model.parameters(), lr = self.learning_rate)
