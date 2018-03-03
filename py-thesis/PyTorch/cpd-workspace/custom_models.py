'''
Colletion of useful models
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

# Example 2-Layer custom network 
class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class LenetZhang(nn.Module):
    def __init__(self):
        super(LenetZhang, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, (5,5), padding=(2,2))  
        self.conv2 = nn.Conv2d(96, 128, (5,5), padding=(2,2))
        self.conv3 = nn.Conv2d(128, 256, (5,5), padding=(2,2))
        self.conv4 = nn.Conv2d(256, 64, (1,1), padding=(0,0))

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.thres = nn.Threshold(0, 1e-6)
    
    def forward(self, x): 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        #print(x.shape)
        x = x.view(-1, torch.numel(x[0]))
        #print(x.shape) 
        x = self.dropout_1(x)
        x = self.fc1(x)
        x = self.thres(x)
        x = self.dropout_2(x)
        x = self.fc2(x)
        return x


# Network defined as CP-Decomposed architecture
class CPD_All_Conv(nn.Module):

    def __init__(self, rank1, rank2, filt_size1, filt_size2, filt_fc1, last_kernel_sz, num_classes):
        super(CPD_All_Conv, self).__init__()

        # 1st layer decomposition
        self.conv11 = nn.Conv2d(3, rank1, 1)
        self.conv12 = nn.Conv2d(rank1, rank1, (3, 1), groups=1)
        self.conv13 = nn.Conv2d(rank1, rank1, (1, 3), groups=rank1)
        self.conv14 = nn.Conv2d(rank1, filt_size1, 1)

        # 2nd layer decomposition
        self.conv21 = nn.Conv2d(filt_size1, rank1, 1)
        self.conv22 = nn.Conv2d(rank1, rank1, (3, 1), groups=rank1)
        self.conv23 = nn.Conv2d(rank1, rank1, (1, 3), groups=rank1)
        self.conv24 = nn.Conv2d(rank1, filt_size2, 1)

        # 3rd layer decomposition
        self.conv31 = nn.Conv2d(filt_size2, rank2, 1)
        self.conv32 = nn.Conv2d(rank2, rank2, (3, 1), groups=rank2)
        self.conv33 = nn.Conv2d(rank2, rank2, (1, 3), groups=rank2)
        self.conv34 = nn.Conv2d(rank2, filt_size2, 1)

        # 4th layer decomposition
        self.conv41 = nn.Conv2d(filt_size2, rank2, 1)
        self.conv42 = nn.Conv2d(rank2, rank2, (3, 1), groups=rank2)
        self.conv43 = nn.Conv2d(rank2, rank2, (1, 3), groups=rank2)
        self.conv44 = nn.Conv2d(rank2, filt_size2, 1)

        # Normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_dropout = nn.Dropout2d(0.25)
        self.bn_1 = nn.BatchNorm2d(rank1)
        self.bn_2 = nn.BatchNorm2d(rank2)
        self.bn_3 = nn.BatchNorm2d(1)
        self.bn_4 = nn.BatchNorm2d(filt_size1)
        self.bn_5 = nn.BatchNorm2d(filt_size2)
        self.bn_6 = nn.BatchNorm2d(filt_fc1)

        # conv2fc
        # 4-way decomposition
        self.cpdfc1 = nn.Conv2d(64, rank1, 1)
        self.cpdfc2 = nn.Conv2d(rank1, rank1, (last_kernel_sz, 1), groups=rank1)
        self.cpdfc3 = nn.Conv2d(rank1, rank1, (1, last_kernel_sz), groups=rank1)
        self.cpdfc4 = nn.Conv2d(rank1, filt_fc1, 1)

        # self.conv2fc1 = nn.Conv2d(64, self.filt_fc1, 5)
        self.conv2fc2 = nn.Conv2d(filt_fc1, num_classes, 1)
        self.fc3 = nn.Linear(10, 10)
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x):
        num_classes = 10

        x = self.conv11(x)
        x = self.bn_1(x)
        x = self.conv12(x)
        x = self.bn_1(x)
        x = self.conv13(x)
        x = self.bn_1(x)
        x = F.relu(self.conv14(x))
        x = self.bn_4(x)

        x = self.conv21(x)
        x = self.bn_1(x)
        x = self.conv22(x)
        x = self.bn_1(x)
        x = self.conv23(x)
        x = self.bn_1(x)
        x = self.pool(F.relu(self.conv24(x)))
        x = self.conv2_dropout(x)

        x = self.conv41(x)
        x = self.bn_2(x)
        x = self.conv42(x)
        x = self.bn_2(x)
        x = self.conv43(x)
        x = self.bn_2(x)
        x = self.pool(F.relu(self.conv44(x)))
        x = self.bn_5(x)

        # x = F.relu(self.conv2fc1(x))
        x = self.cpdfc1(x)
        x = self.bn_1(x)
        x = self.cpdfc2(x)
        x = self.bn_1(x)
        x = self.cpdfc3(x)
        x = self.bn_1(x)
        x = F.relu(self.cpdfc4(x))
        x = self.bn_6(x)
        x = self.conv2fc2(x)

        x = x.view(-1, num_classes)  # Flatten! <---
        x = self.fc3(x)
        return x


# Network as defined in Keras
class Keras_Cifar_classic(nn.Module): 
    def __init__(self):
        super(Keras_Cifar_classic, self).__init__() 

        self.kern = 3  # for all layers
        self.filt_size1 = 32
        self.filt_size2 = 64
        self.filt_fc1 = 512
        self.num_classes = 10

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout_1 = nn.Dropout2d(0.25)
        self.dropout_2 = nn.Dropout2d(0.5)

        # fully connected
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)
        self.classifier = nn.Linear(10,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_1(x)

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout_2(x)
        x = x.view(-1, 64 * 6*6)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)

        return x


# Network as in Keras, wt BN but without groups=#inputs
# Hence, it's decomposable only with Tucker but not with Parafac
class Keras_Cifar2(nn.Module):
    def __init__(self, rank1, rank2):
        super(Keras_Cifar2, self).__init__()
        # hyperparams
        self.kern = 3  # for all layers
        self.filt_size1 = 32
        self.filt_size2 = 64
        self.filt_fc1 = 512
        self.num_classes = 10

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout_1 = nn.Dropout2d(0.25)
        # self.dropout_2 = nn.Dropout2d(0.5)

        self.bn_1 = nn.BatchNorm2d(1)
        self.bn_2 = nn.BatchNorm2d(rank1)
        self.bn_3 = nn.BatchNorm2d(rank2)
        self.bn_4 = nn.BatchNorm2d(self.filt_fc1)
        self.bn_5 = nn.BatchNorm2d(self.num_classes)
        self.bn_6 = nn.BatchNorm2d(32)
        self.bn_7 = nn.BatchNorm2d(64)

        # 4-way decomposition
        self.cpdfc1 = nn.Conv2d(64, rank1, 1)
        self.cpdfc2 = nn.Conv2d(rank1, rank1, (6, 1))
        self.cpdfc3 = nn.Conv2d(rank1, rank1, (1, 6))
        self.cpdfc4 = nn.Conv2d(rank1, self.filt_fc1, 1)

        # conv2fc
        #self.conv2fc1 = nn.Conv2d(64, self.filt_fc1, 5)
        self.conv2fc2 = nn.Conv2d(self.filt_fc1, self.num_classes, 1)

        # fully connected
        # self.fc1 = nn.Linear(64 * 6 * 6, 512)
        # self.fc2 = nn.Linear(512, 10)
        self.classifier = nn.Linear(10, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.bn_6(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn_6(x)

        x = F.relu(self.conv3(x))
        x = self.bn_7(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn_7(x)

        x = self.cpdfc1(x)
        x = self.bn_2(x)
        x = self.cpdfc2(x)
        x = self.bn_2(x)
        x = self.cpdfc3(x)
        x = self.bn_2(x)
        x = F.relu(self.cpdfc4(x))
        x = self.bn_4(x)
        x = self.conv2fc2(x)
        x = self.bn_5(x)

        x = x.view(-1, self.num_classes)  # Flatten! <---
        x = self.classifier(x)
        return x


class NIN(nn.Module): 
    def __init__(self):
        super(NIN, self).__init__()

        self.sequential = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 192, 5, 1, padding=2)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(192, 160,1,1, padding=0)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(160, 96, 1, 1, padding=0)),
                ('relu3', nn.ReLU()),
                ('pool1', nn.MaxPool2d(2,2)),
                ('dropout1', nn.Dropout2d(0.5)),
                ('conv4', nn.Conv2d(96, 192, 5, 1, padding=2)),
                ('relu4', nn.ReLU()),
                ('conv5', nn.Conv2d(192, 192, 1, 1, padding=0)),
                ('relu5', nn.ReLU()),
                ('conv6', nn.Conv2d(192, 192, 1, 1, padding=0)),
                ('relu6', nn.ReLU()),
                ('pool2', nn.AvgPool2d(2, 2)),
                ('dropout2', nn.Dropout2d(0.5)),
                ('conv7', nn.Conv2d(192, 192, 3, 1, padding=1)),
                ('relu7', nn.ReLU()),
                ('conv8', nn.Conv2d(192, 192, 1, 1, padding=0)),
                ('relu8', nn.ReLU()),
                ('conv9', nn.Conv2d(192, 192, 1, 1, padding=0)),
                ('relu9', nn.ReLU()),
                ('classifier', nn.Conv2d(192, 10, 1, 1, padding=0)),
                ('pool3', nn.AvgPool2d(8,1)),
            ]))

        def forward(self, x): 
            x = self.sequential(x)
            x = x.view(-1, 10)
            return x





# Network defined as in Keras, with BN and groups 
class Keras_Cifar_Separable(nn.Module):
    def __init__(self, rank1, rank2):
        super(Keras_Cifar_Separable, self).__init__()
        # hyperparams
        self.kern = 3  # for all layers
        self.filt_size1 = 32
        self.filt_size2 = 64
        self.filt_fc1 = 512
        self.num_classes = 10

        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, rank1, 1),
            nn.BatchNorm2d(rank1),
            nn.Conv2d(rank1, rank1, (6, 1), groups=rank1),
            nn.BatchNorm2d(rank1),
            nn.Conv2d(rank1, rank1, (1, 6), groups=rank1),
            nn.BatchNorm2d(rank1),            
            nn.Conv2d(rank1, self.filt_fc1, 1),
            nn.BatchNorm2d(self.filt_fc1),
            nn.Conv2d(self.filt_fc1, self.num_classes, 1),
            nn.BatchNorm2d(self.num_classes)
        )


    def forward(self, x):    
        x = self.sequential(x)
        x = x.view(-1, self.num_classes)  # Flatten! <---
        return x



# Network defined as in Keras, with BN and groups 
class Keras_Cifar_Separable_Old(nn.Module):
    def __init__(self, rank1, rank2):
        super(Keras_Cifar_Separable_Old, self).__init__()
        # hyperparams
        self.kern = 3  # for all layers
        self.filt_size1 = 32
        self.filt_size2 = 64
        self.filt_fc1 = 512
        self.num_classes = 10

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout_1 = nn.Dropout2d(0.25)
        # self.dropout_2 = nn.Dropout2d(0.5)

        self.bn_1 = nn.BatchNorm2d(1)
        self.bn_2 = nn.BatchNorm2d(rank1)
        self.bn_3 = nn.BatchNorm2d(rank2)
        self.bn_4 = nn.BatchNorm2d(self.filt_fc1)
        self.bn_5 = nn.BatchNorm2d(self.num_classes)
        self.bn_6 = nn.BatchNorm2d(32)
        self.bn_7 = nn.BatchNorm2d(64)

        # 4-way decomposition
        self.cpdfc1 = nn.Conv2d(64, rank1, 1)
        self.cpdfc2 = nn.Conv2d(rank1, rank1, (6, 1), groups=rank1)
        self.cpdfc3 = nn.Conv2d(rank1, rank1, (1, 6), groups=rank1)
        self.cpdfc4 = nn.Conv2d(rank1, self.filt_fc1, 1)

        # conv2fc
        #self.conv2fc1 = nn.Conv2d(64, self.filt_fc1, 5)
        self.conv2fc2 = nn.Conv2d(self.filt_fc1, self.num_classes, 1)

        # fully connected
        # self.fc1 = nn.Linear(64 * 6 * 6, 512)
        # self.fc2 = nn.Linear(512, 10)
        self.classifier = nn.Linear(10, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.bn_6(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn_6(x)

        x = F.relu(self.conv3(x))
        x = self.bn_7(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn_7(x)

        x = self.cpdfc1(x)
        x = self.bn_2(x)
        x = self.cpdfc2(x)
        x = self.bn_2(x)
        x = self.cpdfc3(x)
        x = self.bn_2(x)
        x = F.relu(self.cpdfc4(x))
        x = self.bn_4(x)
        x = self.conv2fc2(x)
        x = self.bn_5(x)

        x = x.view(-1, self.num_classes)  # Flatten! <---
        x = self.classifier(x)
        return x


# Network defined as in Keras
class Keras_Cifar_AllConv(nn.Module):
    def __init__(self):
        super(Keras_Cifar_AllConv, self).__init__()
        # hyperparams
        rank1 = 20
        rank2 = 5
        
        self.kern = 3  # for all layers
        self.filt_size1 = 32
        self.filt_size2 = 64
        self.filt_fc1 = 512
        self.num_classes = 10

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout_1 = nn.Dropout2d(0.25)
        # self.dropout_2 = nn.Dropout2d(0.5)

        self.bn_1 = nn.BatchNorm2d(1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.bn_conv4 = nn.BatchNorm2d(64)
        self.bn_4 = nn.BatchNorm2d(self.filt_fc1)
        self.bn_5 = nn.BatchNorm2d(self.num_classes)
        self.bn_6 = nn.BatchNorm2d(32)
        self.bn_7 = nn.BatchNorm2d(64)

        # 4-way decomposition
        # self.cpdfc1 = nn.Conv2d(64, rank1, 1)
        # self.cpdfc2 = nn.Conv2d(rank1, rank1, (6, 1), groups=1)
        # self.cpdfc3 = nn.Conv2d(rank1, rank1, (1, 6), groups=1)
        # self.cpdfc4 = nn.Conv2d(rank1, self.filt_fc1, 1)
        
        # conv2fc
        self.conv2fc1 = nn.Conv2d(64, self.filt_fc1, 6)
        self.conv2fc2 = nn.Conv2d(self.filt_fc1, self.num_classes, 1)

        # fully connected
        # self.fc1 = nn.Linear(64 * 6 * 6, 512)
        # self.fc2 = nn.Linear(512, 10)
        # self.classifier = nn.Linear(10,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn_conv1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn_conv2(x)

        x = F.relu(self.conv3(x))
        x = self.bn_conv3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn_conv4(x)
        
        '''
        x = self.cpdfc1(x)
        x = self.bn_2(x)
        x = self.cpdfc2(x)
        x = self.bn_2(x)
        x = self.cpdfc3(x)
        x = self.bn_2(x)
        x = F.relu(self.cpdfc4(x))
        '''
        x = F.relu(self.conv2fc1(x))
        x = self.bn_4(x)
        x = self.conv2fc2(x)
        x = self.bn_5(x)

        x = x.view(-1, self.num_classes)  # Flatten! <---
        return x
