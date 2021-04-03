import time

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
        dataset_size = len(dataset)
        training_size = np.round(dataset_size*0.9).astype(int)
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset, [training_size, dataset_size-training_size],
            generator=torch.Generator().manual_seed(42) ) 

    def on_fit_start(self):
        if self.add_graph:
            img = self.train_set[0][0].to(self.device)
            self.logger.experiment.add_graph(Forwarder(self.model), torch.unsqueeze(img, dim=0))

    def mean_step(self, batch, batch_indx):
        x, y = batch
        result = self.get_distribution(y).log_prob(x)
        log_prob = torch.mean( result["log_prob"] )
        logs = {key: torch.mean(value) for key, value in result.items()}
        return {"loss": -log_prob, "log": logs}

    def training_step(self, batch, batch_indx):
        return self.mean_step(batch, batch_indx)
    
    def validation_step(self, batch, batch_indx):
        return self.mean_step(batch, batch_indx)

    def training_epoch_end(self, outputs):
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {key+"/train":
            torch.tensor([x["log"][key] for x in outputs]).mean() for key in outputs[0]["log"].keys() if key != "step"}
        tensorboard_logs["step"] = self.current_epoch
        epoch_dictionary = {
            "loss": mean_loss,
            "log": tensorboard_logs
        }
        print("Training Loss ", mean_loss, end='')
        print("epoch", self.current_epoch)
        return epoch_dictionary
        
    def validation_epoch_end(self, val_step_outputs):
        mean_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        tensorboard_logs = {key+"/validation":
            torch.tensor([x["log"][key] for x in val_step_outputs]).mean() for key in val_step_outputs[0]["log"].keys() if key != "step"}
        tensorboard_logs["step"] = self.current_epoch
        epoch_dictionary = {
            "loss": mean_val_loss,
            "log": tensorboard_logs
        }
        print("Validation loss", mean_val_loss)
        return epoch_dictionary
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size = self.batch_size, num_workers=4, drop_last=True)

class LightningDistributionTrainer(LightningTrainer):
    def __init__(self, model, dataset, add_graph=False, learning_rate=.001, batch_size=32):
        super(LightningDistributionTrainer, self).__init__( model, dataset, add_graph, learning_rate, batch_size)
    def get_distribution(self, y):
        return self.model

def distribution_sample(model):
    imglist = [model.sample() for _ in range(16)]
    imglist = torch.clip(torch.cat(imglist, axis=0),0.0,1.0)
    return torchvision.utils.make_grid(imglist, padding=10, nrow=4 )

class LogDistributionSamplesPerEpoch(pl.Callback):
    def __init__(self, filename=None):
        super(LogDistributionSamplesPerEpoch, self).__init__()
        self.filename = filename

    def on_validation_epoch_end(self, trainer, pl_module):
        samples = distribution_sample(pl_module.model)
        pl_module.logger.experiment.add_image("epoch_image", samples, pl_module.current_epoch, dataformats="CHW")
        if self.filename is not None:
            torchvision.utils.save_image(samples, self.filename)

#Potential issue with batch_indx wrapping around epoch.
#using global_step could result in multiple samples if accumulate_grads > 1
class LogDistributionSamplesPerTraining(pl.Callback):
    def __init__(self, every_global_step=1000, filename=None):
        super(LogDistributionSamplesPerTraining, self).__init__()
        self.every_global_step = every_global_step
        self.filename = filename

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (pl_module.global_step % self.every_global_step == 0) and (batch_idx % trainer.accumulate_grad_batches == 0):
            pl_module.eval()
            samples = distribution_sample(pl_module.model)
            pl_module.train()
            pl_module.logger.experiment.add_image("train_image", samples, pl_module.global_step, dataformats="CHW")
            if self.filename is not None:
                torchvision.utils.save_image(samples, self.filename)


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
    
    def sample(self):
        return self.cond_distribution(torch.unsqueeze(self.conditionals, 0)).sample()
