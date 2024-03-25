import torch.nn as nn
import torchvision
import argparse
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CIFAR10Recognizer(nn.Module):
    def __init__(self):
        super(CIFAR10Recognizer, self).__init__()
        self.net = Net()

    def forward(self, digit_image):
        logits = self.net(digit_image)
        return torch.distributions.categorical.Categorical(logits=logits)


parser = argparse.ArgumentParser(description='PyGen CIFAR10 Example')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
ns = parser.parse_args()

transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(ns.datasets_folder, train=True, download=True, transform=transform)
class_labels = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
cifar10_recognizer = CIFAR10Recognizer()
train_dataset, validation_dataset = random_split(dataset, [45000, 5000])
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([callbacks.TBClassifyImagesCallback(tb_writer, "train_images", train_dataset, class_labels),
                       callbacks.TBClassifyImagesCallback(tb_writer, "validation_images", validation_dataset, class_labels),
                       callbacks.TBLogProbCallback(tb_writer, "train_epoch_log_prob"),
                       callbacks.TBAccuracyCallback(tb_writer, "train_accuracy", train_dataset),
                       callbacks.TBAccuracyCallback(tb_writer, "validation_accuracy", validation_dataset)])
train.layer_train(cifar10_recognizer, train_dataset, batch_end_callback=callbacks.TBLogProbCallback(tb_writer, "batch_log_prob"), epoch_end_callback=epoch_end_callbacks)