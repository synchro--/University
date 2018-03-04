# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.

Specifically for ``vision``, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from logger import Logger
from torch.optim import lr_scheduler

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

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


from pytorch_utils import *

# vars
rank1 = 20
rank2 = 5


'''
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
>>> output = m(input)
'''


# Network defined as in Keras
class Keras_Cifar(nn.Module):
    def __init__(self):
        super(Keras_Cifar, self).__init__()
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
        x = self.bn_6(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn_6(x)

        x = F.relu(self.conv3(x))
        x = self.bn_7(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn_7(x)

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

# Tutorial net
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
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set the logger
logger = Logger('./logs')
log_file = 'cifar10_keras.csv'
model_file = 'm_finetuned.pth'
model_weights = 's_finetuned.pth'
retrain = False

from custom_models import *

# net = Keras_Cifar_Separable(20, 5)
# net = Net()
# net = torch.load(model_file)
net = NIN_BN() 

if retrain:
    # load previous model
    net.load_state_dict(torch.load(model_weights))
if not retrain:
    # init log file
    with open(log_file, 'w') as out:
        out.write('Cifar10-Adam-Step,Accuracy,Loss\n')
        out.close

net.cuda()  # GPU

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

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize
# Decay LR by a factor of 0.1 every 10 epochs


loaders = get_train_valid_loader (data_dir='./data', batch_size=32, augment=False,
                                            random_seed=7)
dataloaders = {'train': loaders[0], 'val': loaders[1]}

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
net = train_model(trainloader, net, criterion, optimizer, exp_lr_scheduler, epochs=50)
dump_model_weights(net)

net.train(False)
torch.save(net, "keras_OUTOUT.pth")

test_model_cifar10(testloader, net)

'''
########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

net.train(False)
outputs = net(Variable(images.cuda()))

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
for data in testloader:
    images, labels = data
    # images.cuda(); labels.cuda()
    outputs = net(Variable(images.cuda(), volatile=True))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))



class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda(), volatile=True))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))
''' 
