import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from logger import Logger
from torch.optim import lr_scheduler
import argparse

from dataloaders import dataset  # Load datasets

import matplotlib.pyplot as plt
import numpy as np
from pytorch_utils import *
from training_algo import *
from models.custom_models import *
from torchsummary import summary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--state", type=str, default="")
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--validation", dest="val", action="store_true")
    parser.add_argument('--ranks', nargs='+', dest="ranks",
                        default=[20, 30, 40],
                        help="Specify the rank of each layer")
    args = parser.parse_args()
    return args


args = get_args()

# Set the logger
logger = Logger('./logs')
log_file = 'cifar10_keras.csv'
model_file = 'm_finetuned.pth'
model_weights = 's_trained.pth'
retrain = False

# 1. load datasets
trainloader = dataset.cifar10_trainloader(
    batch_size=32, num_workers=0, augment=True, pin_memory=True)
testloader = dataset.cifar10_testloader(batch_size=32, num_workers=0)


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
# net = Net()
# net = torch.load(model_file)
# net = NIN_BN()
# net = CPD_All_Conv(int(args.ranks[0]), int(args.ranks[1]), int(args.ranks[2]), relu=False)
# net = Keras_Cifar_classic()
net = LenetZhang()
# net = CPD_Zhang(int(args.ranks[0]), int(args.ranks[1]), int(args.ranks[2]), relu=False)

# GPU
if args.model:
    net = torch.load(args.model)
if args.state:
    net.load_state_dict(torch.load(args.state))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

print('###################')
print(summary(net, (3, 32, 32)))
print('Number of trainable params:',
      sum([param.nelement() for param in net.parameters()]))

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

if args.opt == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

else:
    print('Using SGD for smaller improvements...')
    optimizer = optim.SGD(net.parameters(), lr=0.1e-4,
                          momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

########################################################################
# 4. Train the network
#

if args.val:
    loaders = get_train_valid_loader(data_dir='./data', batch_size=32, augment=False,
                                     random_seed=7)
    dataloaders = {'train': loaders[0], 'val': loaders[1]}
    net = train_model_val(dataloaders, net, criterion,
                          optimizer, exp_lr_scheduler, epochs=25)
else:
    dataloaders = {'train': trainloader, 'test': testloader}
    net = train_test_model(dataloaders, net, criterion,
                           optimizer, exp_lr_scheduler, epochs=50)

# dump_model_weights(net)

torch.save(net, "just_TR.pth")
net.train(False)
test_model_cifar10(testloader, net)
