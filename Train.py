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
        result = self.get_distribution(y).log_prob(x)
        log_prob = torch.mean( result["log_prob"] )
        logs = {key: torch.mean(value) for key, value in result.items()}
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

def distribution_sample(model, temperature=1.0):
    imglist = [model.sample(temperature) for _ in range(16)]
    imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
    return torchvision.utils.make_grid(imglist, padding=10, nrow=4 )


class LogSamplesEpochCallback(pl.Callback):
    def __init__(self, filename=None, temperature=1.0):
        super(LogSamplesEpochCallback, self).__init__()
        self.filename = filename
        self.temperature = temperature

    def on_validation_epoch_end(self, trainer, pl_module):
        samples = distribution_sample(pl_module.model, self.temperature)
        pl_module.logger.experiment.add_image("epoch_sample T"+str(self.temperature), samples, pl_module.current_epoch, dataformats="CHW")
        if self.filename is not None:
            torchvision.utils.save_image(samples, self.filename)


class LogSamplesTrainingCallback(pl.Callback):
    def __init__(self, every_global_step=1000, filename=None, temperature = 1.0):
        super(LogSamplesTrainingCallback, self).__init__()
        self.every_global_step = every_global_step
        self.filename = filename
        self.temperature = temperature

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.every_global_step == 0) and (batch_idx % trainer.accumulate_grad_batches == 0):
            pl_module.eval()
            samples = distribution_sample(pl_module.model, self.temperature)
            pl_module.train()
            pl_module.logger.experiment.add_image("train_sample T"+str(self.temperature), samples, pl_module.global_step, dataformats="CHW")
            if self.filename is not None:
                torchvision.utils.save_image(samples, self.filename)


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

# To train a conditional distribution:
# mydist = Train.PyGenBrixModel( 
#     cnn.MultiStageParallelCNNLayer([1, 28, 28], vae.IndependentBernoulliLayer()),
#     [1, 28, 28])
# Train.train( mydist, mnist, batch_size=32, sleep_time=0)
class PyGenBrixModel(nn.Module):

    def __init__(self, distribution, dims):
        super(PyGenBrixModel, self).__init__()
        self.cond_distribution = distribution
        self.dims = dims
        self.conditionals = torch.nn.Parameter(torch.zeros(dims), requires_grad=True )
        
    def log_prob(self, samples):
        return self.cond_distribution(self.conditionals.expand([samples.shape[0], self.dims[0], self.dims[1], self.dims[2]])).log_prob(samples)
    
    def sample(self, temperature=1.0):
        return self.cond_distribution(torch.unsqueeze(self.conditionals, 0)).sample(temperature)

    def mode(self):
        return self.cond_distribution(torch.unsqueeze(self.conditionals, 0)).mode()
