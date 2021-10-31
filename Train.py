import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl


class Forwarder(nn.Module):

    def __init__(self, model):
        super(Forwarder, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.log_prob(x)["log_prob"]


class LightningTrainer(pl.LightningModule):
    def __init__(self, model, dataset, add_graph=False, learning_rate=.001, batch_size=32):
        super(LightningTrainer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.add_graph = add_graph
        self.dataset = dataset
        self.train_set, self.val_set = self.get_datasets()

    def get_datasets(self):
        dataset_size = len(self.dataset)
        training_size = np.round(dataset_size*0.9).astype(int)
        train_set, val_set = torch.utils.data.random_split(
            self.dataset, [training_size, dataset_size-training_size],
            generator=torch.Generator().manual_seed(42) ) 
        return (train_set, val_set)

    def mean_step(self, batch, batch_indx):
        x, y = batch
        ndims = np.prod(x.shape[1:])
        result = self.get_distribution(y).log_prob(x)
        log_prob = torch.mean( result["log_prob"] )
        logs = {key: torch.mean(value)/ndims for key, value in result.items()}
        return {"loss": -log_prob, "log": logs}

    def training_step(self, batch, batch_indx):
        logs = self.mean_step(batch, batch_indx)
        for key, value in logs["log"].items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logs["loss"]
    
    def validation_step(self, batch, batch_indx):
        logs = self.mean_step(batch, batch_indx)
        for key, value in logs["log"].items():
            self.log("validation_"+key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return logs["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size = self.batch_size, num_workers=4, drop_last=True, pin_memory=True)

class LightningDistributionTrainer(LightningTrainer):
    def __init__(self, model, dataset, add_graph=False, learning_rate=.001, batch_size=32):
        super(LightningDistributionTrainer, self).__init__( model, dataset, add_graph, learning_rate, batch_size)
    def get_distribution(self, y):
        return self.model


def logging_distribution_samples(pl_module, model, name, current_step, batch_size, temperature=1.0, filename=None):
    imglist = [model.sample([batch_size], temperature) for _ in range(16//batch_size)]
    imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
    grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=4 )
    pl_module.logger.experiment.add_image(name+"T"+str(temperature), grid_image, current_step, dataformats="CHW")
    if filename is not None:
        torchvision.utils.save_image(grid_image, filename)


class LogSamplesEpochCallback(pl.Callback):
    def __init__(self, filename=None, temperature=1.0):
        super(LogSamplesEpochCallback, self).__init__()
        self.filename = filename
        self.temperature = temperature

    def on_validation_epoch_end(self, trainer, pl_module):
        logging_distribution_samples(pl_module, pl_module.model, "epoch", pl_module.current_epoch, pl_module.batch_size, self.temperature, self.filename)


class LogSamplesTrainingCallback(pl.Callback):
    def __init__(self, every_global_step=1000, filename=None, temperature = 1.0):
        super(LogSamplesTrainingCallback, self).__init__()
        self.every_global_step = every_global_step
        self.filename = filename
        self.temperature = temperature

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.every_global_step == 0) and (batch_idx % trainer.accumulate_grad_batches == 0):
            pl_module.eval()
            logging_distribution_samples(pl_module, pl_module.model, "train", pl_module.global_step, pl_module.batch_size, self.temperature, self.filename)
            pl_module.train()


class LogReconstructionEpochCallback(pl.Callback):
    """LogReconstructionEpochCallback logs images from an autoencoder
    which defines sample( images )
    """
    def __init__(self):
        super(LogReconstructionEpochCallback, self).__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        sample_imglist = [ [
            pl_module.val_set[i*2][0].cuda(),
            pl_module.model.sample_reconstruction( torch.unsqueeze(pl_module.val_set[i*2][0].cuda(),0) )[0],
            pl_module.val_set[i*2+1][0].cuda(),
            pl_module.model.sample_reconstruction( torch.unsqueeze(pl_module.val_set[i*2+1][0].cuda(),0) )[0] ] for i in range(4) ]
        trainer.logger.experiment.add_image("reconstruction_T=1.0", torchvision.utils.make_grid(list(itertools.chain(*sample_imglist)), padding=10, nrow=4 ), pl_module.global_step, dataformats="CHW")


#To run a training session:
#pl.Trainer(fast_dev_run=False, gpus=1).fit(Train.LightningTrainer(mymodel, dataset, Train.disp, batch_size=16))
#To restore a training session:
#_ = Train.LightningTrainer.load_from_checkpoint("~/PyGenBrixProj/CelebA/lightning_logs/version_7/checkpoints/epoch=1-step=22793.ckpt", model=mymodel, dataset=celeba_dataset)
