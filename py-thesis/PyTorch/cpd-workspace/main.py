import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain

from decompositions import *
from pytorch_utils import *
from training_algo import *
from torch.optim import lr_scheduler
from custom_models import *

import subprocess

# hyperparams
rank1 = 20
rank2 = 5

kern = 3  # for all layers
last_kern = 6
filt_size1 = 32
filt_size2 = 64
filt_fc1 = 512
num_classes = 10

# VGG16 based network for classifying between dogs and cats.
# After training this will be an over parameterized network,
# with potential to shrink it.
class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self, model=None):
        super(ModifiedVGG16Model, self).__init__()

        model = models.VGG16(pretrained=True)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Trainer:
    def __init__(self, train_path, test_path, model, optimizer):
        # self.train_data_loader = dataset.loader(train_path)
        # self.test_data_loader = dataset.test_loader(test_path)
        # self.train_data_loader = dataset.cifar10_trainloader()
        self.train_data_loader, self.valid_data_loader = get_train_valid_loader(data_dir='./data',
         batch_size=32, augment=False, random_seed=7)
        self.test_data_loader = dataset.cifar10_testloader()

        self.optimizer = optimizer

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

    def test(self):
        self.model.cuda()
        # self.model.eval()
        correct = 0
        total = 0
        total_time = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = model(Variable(batch)).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy : %.4f" % (float(correct) / total))
        print("Average prediction time %.4f %d" % (float(total_time) / (i + 1), i + 1))

        self.model.train()

    def train(self, epoches=10):
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch()
            self.test()
        print("Finished fine tuning.")


    def train_batch(self, batch, label):
        self.model.zero_grad()
        input = Variable(batch)
        self.criterion(self.model(input), Variable(label)).backward()
        self.optimizer.step()

    def train_epoch(self):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(batch.cuda(), label.cuda())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = "cifar")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--full-decompose", dest="full_decompose", action="store_true")
    parser.add_argument("--layer",type=str, default="conv1")
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default")
    parser.add_argument('--layers', nargs='+', dest="layers",
                            default=['conv4', 'conv3', 'conv2', 'conv1'], 
                            help="Specify the names of the layers to decompose")
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    args = parser.parse_args()
    return args



def decompose_model(model, layer_name, model_file):
    print(model)
    model.cpu()
    for i, (name, conv_layer) in enumerate(model.named_modules()):
        ## for sequential nets, 'in' is sufficient
        ## as long as there are not 2 homonimous layers
        if layer_name in name: 
            print(name)
            if args.cp:
                # rank = max(conv_layer.weight.data.shape) // 3
                rank = cp_ranks(conv_layer)
                print('rank: ', rank)
                if 'conv2fc' in layer_name:
                    rank = 40
                decomposed = cp_decomposition_conv_layer_BN(conv_layer, rank)
                # decomposed = cp_xavier_conv_layer(conv_layer, rank)
            else:
                decomposed = tucker_decomposition_conv_layer(conv_layer)

    model._modules[layer_name] = decomposed

    torch.save(model, model_file)
    return model


if __name__ == '__main__':
    args = get_args()
    tl.set_backend('numpy')

    if args.train:
        model = ModifiedVGG16Model().cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.train(epoches = 10)
        torch.save(model, "model")

    # Decompose all the specified layers without fine-tuning 
    # Save the architecture in "full_decomposed.pth"
    elif args.full_decompose: 
        layers = args.layers
        model = LenetZhang()
        model.load_state_dict(torch.load(args.model))
        for i, layer in enumerate(layers):
            dec = decompose_model(model, layer, 'decomposed_model.pth')
            for param in dec.parameters():
                param.requires_grad = True
            print(torch_summarize(dec))
            print('###################')
            print('Number of trainable params:', sum([param.nelement() for param in dec.parameters()]))
        torch.save(dec, "full_decomposed.pth")

    
    elif args.fine_tune:
        # 1st time decomposition
        model = LenetZhang()
        # model = torch.load('decomposed_model.pth')
        # model = torch.load('full_decomposed.pth')
        model.load_state_dict(torch.load(args.model))
        # model = torch.load('LAST-tucker.pth')
        # model = torch.load('finetuned.pth') 
        print(torch_summarize(model))
        
        layers = args.layers

        for i, layer in enumerate(layers):
            dec = decompose_model(model, layer, 'decomposed_model.pth')
            dec = model 
            dec.cuda()

            for param in dec.parameters():
                param.requires_grad = True
            print(torch_summarize(dec))
            print('###################')
            print('Number of trainable params:', sum([param.nelement() for param in dec.parameters()]))

            ### Training with my training procedure ###
            if args.cp:
                lr = 0.001
                step_size = 30
            else:
                lr = 0.001
                step_size = 35

            optimizer = optim.Adam(dec.parameters(), lr=lr)
            # Decay LR by a factor of 0.1 every X epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
            
            # loaders = get_train_valid_loader (data_dir='./data', batch_size=32, augment=False, random_seed=7)
            #dataloaders = {'train': loaders[0], 'val': loaders[1]}
            
            criterion = torch.nn.CrossEntropyLoss()
            
            dec = train_model(dataset.cifar10_trainloader(), dec, criterion,
                            optimizer, exp_lr_scheduler, 0.276, 50)
            
            
           # train_model_val(dataloaders, dec, criterion, 
           #                     optimizer, exp_lr_scheduler, 25)
            
            test_model_cifar10(dataset.cifar10_testloader(), dec)
            
            if args.cp:
                logname = "cp-reverse." + str(layer) + ".csv"
                modelname = "s_cp-reverse." + str(layer) + ".pth"
                cmd1 = "mv cifar10.csv " + logname
                cmd2 = "cp s_trained.pth " + modelname
            else:
                logname = "tucker-reverse." + str(layer) + ".csv"
                modelname = "s_tucker-reverse." + str(layer) + ".pth"
                cmd1 = "mv cifar10.csv " + logname
                cmd2 = "cp s_trained.pth " + modelname

            subprocess.call(cmd1.split())
            subprocess.call(cmd2.split())
            
            # Save last model
            torch.save(dec, 'finetuned.pth')



