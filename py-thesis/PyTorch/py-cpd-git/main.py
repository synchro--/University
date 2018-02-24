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
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer


from pytorch_utils import *
from training_algo import * 
from torch.optim import lr_scheduler
from custom_models import Keras_Cifar

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
        self.train_data_loader = dataset.cifar10_trainloader()
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
    parser.add_argument("--layer",type=str, default="conv1") 
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default") 
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    tl.set_backend('numpy')
    
    if args.train:
        model = ModifiedVGG16Model().cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.train(epoches = 10)
        torch.save(model, "model")

    elif args.decompose:
        # 1st time decomposition 
        # model = Keras_Cifar(rank1=20, rank2=5)
        # model.load_state_dict(torch.load(args.model))
        # 2 or more layers decomposed 
        model_file = "decomposed_model.pth"
        # model_state = 
        model = torch.load(model_file)
        model.load_state_dict(torch.load("s_trained.pth"))
        print(model)
        model.cpu() # name of the model 
        model.eval()
        

        # N = len(model._modules.keys())
         
        layer_name = args.layer

        for i,key in enumerate(model._modules.keys()): 
            if key == layer_name:
                conv_layer = model._modules[key] 
                if args.cp: 
                    rank = max(conv_layer.weight.data.shape) // 3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)
                
                model._modules[key] = decomposed
         
        torch.save(model, 'decomposed_model.pth')

        '''    
        N = len(model.features._modules.keys())
        for i, key in enumerate(model.features._modules.keys()):

            if i >= N - 2:
                break
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                if args.cp:
                    rank = max(conv_layer.weight.data.numpy().shape)//3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)

                model.features._modules[key] = decomposed
        '''
    elif args.fine_tune:
        base_model = torch.load("decomposed_model.pth")
        # model = torch.nn.DataParallel(base_model)
        # base_model.load_state_dict(torch.load('saved_models/s_cp-cl-all.pth'))
        model = base_model

        for param in model.parameters():
            param.requires_grad = True

        print(torch_summarize(model))
        print('###################')
        print('Number of trainable params:', sum([param.nelement() for param in model.parameters()]))

        model.cuda()    
            
        #-----------------------------------------#
        ### Training with my training procedure ###

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Decay LR by a factor of 0.1 every X epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
        trainloader = dataset.cifar10_trainloader()
        criterion = torch.nn.CrossEntropyLoss()
        model = train_model(trainloader, model, criterion,
                            optimizer, exp_lr_scheduler, 50)
        
        test_model_cifar10(dataset.cifar10_testloader(), model)
        
        if args.cp:
            optimizer = optim.SGD(model.parameters(), lr=0.000001)
        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            optimizer = optim.SGD(model.parameters(), lr=0.001)
        
        #----------------------------------------#
        ### Fine-tuning with the Trainer class ###

        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.test()
        model.cuda()
        model.train()
        trainer.train(epoches=5)
        model.eval()
        trainer.test()

        torch.save(model, 'finetuned.pth')

