import torch
from torch import nn
from sys import exit
from torchinfo import summary

# Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
      # conv1
      x = self.relu(self.conv1(x))
      x = self.pool(x)
      # conv2
      x = self.relu(self.conv2(x))
      x = self.pool(x)
      # conv3
      x = self.relu(self.conv3(x))
      x = x.reshape((x.shape[0], -1))

      # classifier
      x = self.relu(self.linear1(x))
      x = self.linear2(x)
      return x


# Toy data
x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(summary(model, [64, 1, 32, 32]))