import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from logger import Logger
from torch.optim import lr_scheduler
import argparse

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np
from pytorch_utils import *
from training_algo import *
from custom_models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--state", type=str, default="")
    parser.add_argument("--validation", dest="val", action="store_true")
    args=parser.parse_args()
    return args

args = get_args()

# Set the logger
logger = Logger('./logs')
log_file = 'cifar10_keras.csv'
model_file = 'm_finetuned.pth'
model_weights = 's_trained.pth'
retrain = False


# net = Keras_Cifar_Separable(20, 5)
# net = Net()
# net = torch.load(model_file)
# net = NIN_BN() 
# net = CPD_All_Conv(relu=False)
# net.xavier_init()
# net = Keras_Cifar_classic()
# 
# GPU

if args.model: 
    net = torch.load(args.model)
if args.state: 
    net.load_state_dict(torch.load(args.state))


net.cuda() 

print(torch_summarize(net))
print('###################')
print('Number of trainable params:',
      sum([param.nelement() for param in net.parameters()]))

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
optimizer = optim.Adam(net.parameters(), lr=0.001)

########################################################################
# 4. Train the network
# 


loaders = get_train_valid_loader (data_dir='./data', batch_size=32, augment=False,
                                            random_seed=7)
dataloaders = {'train': loaders[0], 'val': loaders[1]}

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if args.val: 
    net = train_model_val(dataloaders, net, criterion, optimizer, exp_lr_scheduler, epochs=25)
else: 
    net = train_model(trainloader, net, criterion, optimizer, exp_lr_scheduler, epochs=25)

# dump_model_weights(net)

net.train(False)
torch.save(net, "just_TR.pth")
test_model_cifar10(testloader, net)
