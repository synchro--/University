## Test model on cifar10 test set 

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from training_algo import test_model_cifar10
from custom_models import *
import sys 

def cifar10_testloader(root='./data', batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    return data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


model = Keras_Cifar2(20, 5)

model_weights = sys.argv[1]
model.load_state_dict(torch.load(model_weights))
model.cuda()

print("model loaded. Now testing...")

test_model_cifar10(cifar10_testloader(), model)