# Training using torchsample functionalities 

import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain

# Torchsample
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy

# My libs 
from decompositions import *
from pytorch_utils import *
from training_algo import *
from custom_models import *

import subprocess

# 0. Args 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--state", type=str, default="")
    parser.add_argument("--validation", dest="val", action="store_true")
    args = parser.parse_args()
    return args

args = get_args()

# 1. Load datasets 

 loaders = get_train_valid_loader (data_dir='./data', batch_size=32, augment=False, random_seed=7)
 dataloaders = {'train': loaders[0], 'val': loaders[1]}

# 2. Create or Load Model 

if args.model:
    model = torch.load(args.model)
if args.state:
    model.load_state_dict(torch.load(args.state))

model = LenetZhang()

## Model Summary 
print(torch_summarize(model))
print('###################')
print('Number of trainable params:',
      sum([param.nelement() for param in model.parameters()]))

#####################################################################
###  3. Define a Loss function and optimizer


import torch.optim as optim
from torch.optim import lr_scheduler

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

########################################################################
# 4. Train the network

trainer = ModuleTrainer(model)

#define callbacks 
callbacks = [EarlyStopping(patience=7),
             ReduceLROnPlateau(factor=0.1, patience=5)]
#regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
#                L2Regularizer(scale=1e-5, module_filter='fc*')]

#constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*')]

initializers = [XavierUniform(bias=False, module_filter='conv*')]
metrics = [CategoricalAccuracy(top_k=1)]

trainer.compile(loss='nll_loss',
                optimizer=optimizer,
                initializers=initializers,
                metrics=metrics)

#summary = trainer.summary([1,28,28])
#print(summary)

trainer.fit(x_train, y_train,
            val_data=(x_test, y_test),
            num_epoch=20,
            batch_size=128,
            verbose=1)
