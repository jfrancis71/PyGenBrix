import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        kernel1 = torch.tensor([[-1,0,+1],[-2,0,+2],[-1,0,+1]]).repeat(1,1,1).type(torch.float)
        kernel2 = torch.tensor([[-1,-2,-1],[0,0,0],[+1,+2,+1]]).repeat(1,1,1).type(torch.float)
        self.kernel = torch.stack([kernel1,kernel2])
        self.conv1 = torch.nn.Conv2d(2,16,5)

    def forward(self, x):  # input x of form Batchx32x32
        x = F.conv2d(x.unsqueeze(1), self.kernel)
        x = (x>.2)*1.0
        x = (self.conv1(x)>.2)*1.0
        x = torch.nn.MaxPool2d([1,4], stride=2)(x)
        return x
