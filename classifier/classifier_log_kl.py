import torch
import torch.nn as nn
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

def train(model, train_loader, optimizer, epoch):
    global global_idx
    model.train()
    total_loss = 0.0
    total_kl_div = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        log_probs = model(data)
        conditional_distribution = torch.distributions.categorical.Categorical(logits=log_probs)
        source_distribution = torch.distributions.categorical.Categorical(logits=log_probs.detach())
        loss = -conditional_distribution.log_prob(target).mean()
        loss.backward()
        optimizer.step()
        global_idx += 1
        log_probs = model(data)
        new_distribution = torch.distributions.categorical.Categorical(logits=log_probs.detach())
        kl_div = torch.distributions.kl.kl_divergence(source_distribution, new_distribution)
        total_loss += loss.item()
        total_kl_div += kl_div.mean().item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL_Div: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss/100, total_kl_div/100))
            tb_writer.add_scalar("mean_loss", total_loss/100, global_idx)
            tb_writer.add_scalar("mean_kl_div", total_kl_div / 100, global_idx)
            total_loss = 0.0
            total_kl_div = 0.0


def test(model, loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    tb_writer.add_scalar("correct", 100. * correct / len(loader.dataset), epoch)

ap = argparse.ArgumentParser(description='Classifier')
ap.add_argument("--dataset_folder")
ns = ap.parse_args()

transform=transforms.Compose([
    transforms.ToTensor()
    ])
train_dataset, valid_dataset = torch.utils.data.random_split(
    datasets.CIFAR10(ns.dataset_folder, train=True, download=False,
                       transform=transform),
    [.9, .1])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=.001)

tb_writer = SummaryWriter("./logs")
global_idx = 0

for epoch in range(1, 10):
    train(model, train_loader, optimizer, epoch)
    test(model, valid_loader, epoch)