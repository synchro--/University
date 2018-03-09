'''
A personal collection of PyTorch utils for Deep Learning.
To split eventually into different modules
A. Salman 
'''


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


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def log_csv(step, acc, loss, file='cifar10.csv'):
    with open(file, 'a') as out:
        out.write("%d,%.3f,%.3f\n" % (step, acc, loss))
        out.close()


def log_compression(layer_weights, compression_factor, file='compression.txt'):
    with open(file, 'a') as out:
        out.write("Weights before: %d - Weights after:%d - Compression ratio: %.4f\n" %
                  (layer_weights.size, (layer_weights.size / compression_factor), compression_factor))
        out.close()


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
        layer: the specified layer 
        tensor: tensor as an ndarray (Numpy)
    '''
    if not(layer.weight.data.numpy().shape == tensor.shape):
        raise Exception('Size mismatch! Cannot asssign weights')

    layer.weight.data = torch.from_numpy(np.float32(tensor))

# Summary of a model, as in Keras .summary() method


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and
        # weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

 # ============ TensorBoard logging ============#


def tensorboard_log(steps, model, info, dir='./logs'):
    logger = Logger(dir)

    for tag, value in info.items():
        logger.scalar_summary(tag, value, steps)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in model.named_parameters():
        # print(str(tag)+"  "+str(value))
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), steps)
        if 'bn' not in tag and value.grad is not None:
            logger.histo_summary(
                tag + '/grad', to_np(value.grad), steps)


# Helper function to save weights in MAT format
def save_weigths_to_mat(allweights, save_dir):
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

        f_last  = cpd[0][0]
        f_first = cpd[0][1]
        f_vertical = cpd[0][2]
        f_horizontal = cpd[0][3]
        print('Loaded cpd weights succesfully.')

        return f_last, f_first, f_vertical, f_horizontal   #  , bias



# def save_best_model(best_avg, current_avg, )

def xavier_weights(self): 
    for m in self.modules(): 
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight)

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
        layer.bias.data_zero_()
    else:
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))


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
