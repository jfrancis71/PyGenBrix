import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import PyGenBrix.models.parser
import pytorch_cifar.models.resnet as resnet
import pytorch_cifar.models.densenet as densenet
import pytorch_cifar.models.preact_resnet as resnetv2
import pytorch_cifar.models.resnext as resnext
import pytorch_cifar.models.lenet as lenet
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.net = model
        self.automatic_optimization = False

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        global running_tot
        x, y = batch
        x = x - .5
        self.optimizers().zero_grad()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        self.optimizers().step()
        tensorboard_logs = {'train_loss': loss}
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x -.5
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(y.view_as(pred)).sum()
        return {'val_loss': loss, "correct":correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        self.log("validation_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valiation_correct", correct/5000, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.001)


ap = argparse.ArgumentParser(description="classifier")
ap.add_argument("--model", default="Classifier")
ap.add_argument("--max_epochs", default=10, type=int)
ap.add_argument("--tensorboard_log")
ns = ap.parse_args()

if ns.model=="lenet":
    model = lenet.LeNet()
if ns.model=="resnet":
    model = resnet.ResNet18()
elif ns.model=="resnetv2":
    model = resnetv2.PreActResNet18()
elif ns.model=="densenet":
    model = densenet.DenseNet121()
elif ns.model=="resnext":
    model = resnext.ResNeXt29_2x64d()
lit_model = Classifier(model)
dataset = datasets.CIFAR10(root='/home/julian/ImageDataSets/CIFAR10', train=True, download=False,
    transform=transforms.Compose([transforms.ToTensor()]))
dataset_size = len(dataset)
print("Dataset size=", dataset_size)
training_size = np.round(dataset_size*0.9).astype(int)
train_set, val_set = torch.utils.data.random_split(
    dataset, [training_size, dataset_size-training_size],
    generator=torch.Generator().manual_seed(42) ) 

cifar_train = DataLoader(train_set, batch_size=32, num_workers=4)
cifar_val = DataLoader(val_set, batch_size=32, num_workers=4)

trainer = pl.Trainer(gpus=1, max_epochs=ns.max_epochs, default_root_dir=ns.tensorboard_log)
trainer.fit(lit_model, cifar_train, cifar_val)
