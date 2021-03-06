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

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy

from pytorch_utils import *
from training_algo import *

import matplotlib.pyplot as plt
import numpy as np
from logger import Logger
from torch.optim import lr_scheduler

#
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#
# Let us show some of the training images, for fun.
# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next() # qui prende BATCHES immagini

# This prints batch size, i.e. 32 
# print('Size: images: {:.2f}'.format(len(images)))
# This prints dataset size, i.e. 50K 
# print('Size: Data: {:.2f}'.format(len(trainset)))

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


log_file = 'cifar10.csv'
model_file = 'cifar10.pth'

# hyperparams
rank1 = 20
rank2 = 5

kern = 3  # for all layers
last_kern = 6
filt_size1 = 32
filt_size2 = 64
filt_fc1 = 512
num_classes = 10


# Network defined as CP-Decomposed architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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
        self.cpdfc2 = nn.Conv2d(rank1, rank1, (last_kern, 1), groups=rank1)
        self.cpdfc3 = nn.Conv2d(rank1, rank1, (1, last_kern), groups=rank1)
        self.cpdfc4 = nn.Conv2d(rank1, filt_fc1, 1)
        # self.cpd2 =

        # self.conv2fc1 = nn.Conv2d(64, self.filt_fc1, 5)
        self.conv2fc2 = nn.Conv2d(filt_fc1, num_classes, 1)
        self.fc3 = nn.Linear(10, 10)
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x):
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



# Set the logger
logger = Logger('./logs')
retrain = False

net = Net()
if retrain:
    # load previous model
    net.load_state_dict(torch.load(model_file))
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

#
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

net = train_model(trainloader, net, criterion, optimizer, exp_lr_scheduler)

'''
#
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize

total_step = 0
best_loss = 100.0
best_model = False

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        # _, argmax = torch.max(outputs, 1)
        # accuracy = (labels == argmax.squeeze()).float().mean()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 999:  # print every 2000 mini-batches
            avg_loss = running_loss / 1000
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                            avg_loss))

            total_step += 1000

            #============ Logging ============#
            # (1) Log the scalar values
            info = {
                # 'loss': loss.data[0],
                'loss': avg_loss,
                # 'accuracy': accuracy.data[0]
            }

            #my logs
            log_csv(total_step, 5, info['loss'])
            tensorboard_log(logger, net, info)

            # for each epoch, save best model
            if best_loss > avg_loss and (i + 1) == 12000:
                print('loss improved from %.3f to %.3f'
                      % (best_loss, avg_loss))
                print('Saving model to ' + model_file + "...\n")
                best_loss = avg_loss
                torch.save(net.state_dict(), model_file)

            # Reset running loss for next iteration
            running_loss = 0.0

print('Finished Training')
'''

#
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

#
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(Variable(images.cuda()))

#
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
for data in testloader:
    images, labels = data
    # images.cuda(); labels.cuda()
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor on to the GPU, you transfer the neural
# net onto the GPU.
# This will recursively go over all modules and convert their parameters and
# buffers to CUDA tensors:
#
# .. code:: python
#
#     net.cuda()
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ::
#
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is realllly small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: http://pytorch.slack.com/messages/beginner/
