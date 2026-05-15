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
from pytorch_utils import * 

def cifar10_testloader(root='./data', batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    return data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


#model = Keras_Cifar_AllConv()
#model = LenetZhang() 
model = Keras_Cifar_classic() 
# model = CPD_All_Conv(relu=False) 

if len(sys.argv) == 3:
    print("Loading model from " + str(sys.argv[2])) 
    model = torch.load(sys.argv[2])

model_weights = sys.argv[1]
model.load_state_dict(torch.load(model_weights))
model.cpu()
model.eval()

print(torch_summarize(model)) 
print('Number of trainable params: ',sum([param.nelement() for param in model.parameters()])) 

print("model loaded. Now testing...")
test_model_cifar10(cifar10_testloader(), model)
