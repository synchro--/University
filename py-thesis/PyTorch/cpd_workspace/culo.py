import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from logger import Logger
from torch.optim import lr_scheduler
import argparse

import dataset # Load datasets

import matplotlib.pyplot as plt
import numpy as np
from pytorch_utils import *
from training_algo import *
from custom_models import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Keras_Cifar_Separable(20, 5)
net = Net()
# net = torch.load(model_file)
# net = NIN_BN()
# net = CPD_All_Conv(int(args.ranks[0]), int(args.ranks[1]), int(args.ranks[2]), relu=False)
# net = Keras_Cifar_classic()
# net = LenetZhang()
# net = CPD_Zhang(int(args.ranks[0]), int(args.ranks[1]), int(args.ranks[2]), relu=False)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

print('###################')
print(summary(net, (3, 32, 32)))
print('Number of trainable params:',
      sum([param.nelement() for param in net.parameters()]))

