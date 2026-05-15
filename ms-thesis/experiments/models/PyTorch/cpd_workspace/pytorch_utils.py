'''
A personal collection of PyTorch utils for Deep Learning.
To split eventually into different modules.
---------
A. Salman
'''
# pytorch
from torch.nn.modules.module import _addindent
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Utils
import numpy as np
import matplotlib.pyplot as plt
from logger import Logger
import scipy.io as sio
import os
import time
import math
import sys
import json
import shutil


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Set of utils taken from CS231 Stanford Class


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Saving file to: " + filepath)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

 # ============ TensorBoard logging ============ #
 # da eliminare


def tensorboard_log(steps, model, info, dir='./logs'):
    logger = Logger(dir)

    for tag, value in info.items():
        logger.scalar_summary(tag, value, steps)

    # (2) Log values and gradients of the parameters (histogram)
    '''
    for tag, value in model.named_parameters():
        # print(str(tag)+"  "+str(value))
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), steps)
        if 'bn' not in tag and value.grad is not None:
            logger.histo_summary(
                tag + '/grad', to_np(value.grad), steps)
    '''


'''
def log_csv(step, acc, loss, val=0, file='cifar10.csv'):
    with open(file, 'a') as out:
        out.write("%d,%.3f,%.3f\n" % (step, acc, loss))
        out.close()


def log_test(step, val=0, file='test.csv'):
    with open(file, 'a') as out:
        out.write("%d,%.3f\n" % (step, val))
        out.close()

def log_compression(layer_weights, compression_factor, file='compression.txt'):
    with open(file, 'a') as out:
        out.write("Weights before: %d - Weights after:%d - Compression ratio: %.4f\n" %
                  (layer_weights.size, (layer_weights.size / compression_factor), compression_factor))
        out.close()
'''


def get_layer_bias(layer, numpy=True):
    '''
        Return weights of the given layer.
    Args:
        numpy: bool. Defalt true. If false return weights as torch array.
    '''

    print('Retrieving weights of size: ' +
          str(layer.bias.data.cpu().numpy().shape))
    if numpy:
        return layer.bias.data.cpu().numpy()
    else:
        return layer.bias


def set_layer_bias(layer, tensor):
    '''
    Set specified tensor as the layer weights. If sizes don't match raise exception.
    Args:
        layer: the specified layer
        tensor: tensor as an ndarray (Numpy)
    '''
    if not(layer.bias.numpy().shape == tensor.shape):
        raise Exception('Size mismatch! Cannot asssign weights')

    layer.bias = torch.from_numpy(np.float32(tensor))


def get_layer_weights(layer, numpy=True):
    '''
        Return weights of the given layer.
    Args:
        numpy: bool. Defalt true. If false return weights as torch array.
    '''

    print('Retrieving weights of size: ' +
          str(layer.weight.data.cpu().numpy().shape))
    if numpy:
        return to_np(layer.weight)
    else:
        return layer.weight.data


def set_layer_weights(layer, tensor):
    '''
    Set specified tensor as the layer weights. If sizes don't match raise exception.

    Args:
        layer: (torch.nn.Module) the specified layer
        tensor: (numpy array) tensor as an ndarray (Numpy)
    '''
    if not(layer.weight.data.numpy().shape == tensor.shape):
        print(layer.weight.data.numpy().shape)
        print(tensor.shape)
        raise Exception('[MY]: Size mismatch! Cannot assign weights')

    layer.weight.data = torch.from_numpy(np.float32(tensor))


###################################################
# HELPER TO SAVE/LOAD .mat files to use with
# Matlab Tensorlab toolbox
# TODO: integrate it in the decomposer class

# Helper function to save weights in MAT format
def save_weigths_to_mat(allweights, save_dir):
    """
    Helper function to save model weights to .mat files 
    So that they can be used inside Maltab tensor toolboxes like 
    Tensorlab. 

    Args:
        allweights: (list) a list in which each member is itself a list containing a pair of values [weights, bias] for each layer
        save_dir: (string) directory to save the files. 
    """
    for idx, weights in enumerate(allweights):
        name = os.path.join(save_dir, "conv" + str(
            idx) + ".mat")  # conv1.mat, conv2.mat, ...
        sio.savemat(name,  {'weights': weights})


def dump_model_weights(model, save_dir='./dumps'):
    '''
    Dump weights for all Conv2D layers and saves it as .mat files
    TODO: Add check if file exists
    '''
    save_dir = os.path.join(os.getcwd(), save_dir)
    # create dir if not exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    allweights = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            print('Saving layer: ' + str(layer) + ' to ' + save_dir)
            tmp = []
            tmp.append(layer.weight)
            tmp.append(layer.bias)
            allweights.append(tmp)

    save_weigths_to_mat(allweights, save_dir)


def dump_layer_weights(layer, filename="weights.mat", save_dir='dumps/'):
    '''
    Dump weights for specified layer as .mat file
    '''
    save_dir = os.path.join(os.getcwd(), save_dir)
    # create dir if not exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    weights = get_layer_weights(layer, numpy=True)
    print('Saving layer ' + str(layer) + " to" + save_dir)

    name = save_dir + filename
    print(name)
    sio.savemat(name,  {'weights': weights})


def load_cpd_weights(filename):
    import scipy.io as sio
    import os

    if not os.path.isfile(filename):
        print("ERROR: .mat file not found")
        return

    # load struct 'cpd_s' from file
    mat_contents = sio.loadmat(filename)['cpd_s']

  #  bias = mat_contents['bias'][0][0][0]  # retrieve bias weights
    cpd = mat_contents['weights'][0][0]  # cell of 4 tensors

    f_last = cpd[0][0]
    f_first = cpd[0][1]
    f_vertical = cpd[0][2]
    f_horizontal = cpd[0][3]
    print('Loaded cpd weights succesfully.')

    return f_last, f_first, f_vertical, f_horizontal  # , bias


# def save_best_model(best_avg, current_avg, )

# this works
def xavier_init_layer(layer):
    if isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    else:
        torch.nn.init.xavier_uniform(layer.weight)


def xavier_init_net(self):
    for m in self.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        else:
            torch.nn.init.xavier_uniform(m.weight)


'''
Init layer paramaters,
see: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
'''


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


## Progress Bar vars ##

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

# xlua-like progress bar


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


'''
# Xavier init for custom NN modules
def xavier_init_net(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def xavier_init(layer):
    if isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    else:
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))
'''

"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
