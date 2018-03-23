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

trainloader = dataset.cifar10_trainloader(augment=False)
testloader = dataset.cifar10_testloader()

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
net = Keras_Cifar_classic()

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

optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

########################################################################
# 4. Train the network
# 

if args.val: 
    loaders = get_train_valid_loader(data_dir='./data', batch_size=32, augment=False,
                                    random_seed=7)
    dataloaders = {'train': loaders[0], 'val': loaders[1]}
    net = train_model_val(dataloaders, net, criterion, optimizer, exp_lr_scheduler, epochs=25)
else: 
    dataloaders = {'train': trainloader, 'test': testloader}
    net = train_test_model(dataloaders, net, criterion, optimizer, exp_lr_scheduler, epochs=50)

# dump_model_weights(net)

net.train(False)
torch.save(net, "just_TR.pth")
test_model_cifar10(testloader, net)
