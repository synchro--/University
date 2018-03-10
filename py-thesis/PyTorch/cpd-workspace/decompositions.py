import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF

from pytorch_utils import *

# Returns tuned ranks according to the desired compression factor
def choose_compression(layer, ranks, compression_factor=2, flag='Tucker2'):
    '''
    Compute tuned ranks according to the desired compression 
    factor. Sometimes VBMF returns too large ranks; hence the 
    decomposition makes the layer bigger instead of shrinking it. 
    This function prevents it. 
    N.B. by default, if the compression is higher than 2
    the ranks will be untouched.

    Args:
        layer: the layer to be compressed
        ranks : estimated ranks 
        compression_factor: how much the layer will compressed
        flag: string, choose compression over different decompositions. 
              default is Tucker. 
    Returns: 
        the newly estimated rank according to desired compression
    '''
    # format is [OUT, IN, k1, k2]
    weights= layer.weight.data.numpy()
    T = weights.shape[0]
    S = weights.shape[1]
    d = weights.shape[2]

    if flag == 'Tucker2':
        compression = ((d**2)*S*T) / ((S*ranks[0] + ranks[0]*ranks[1] * (d**2) + T*ranks[1]) )
        ranks[0] = ranks[0] * 3
        print(compression)


        '''
        if compression == 2.812156719150494:
            ranks[1] = 13 
            ranks[0] = 49 
            return ranks 
        '''
        # compression must be 2 or more, otherwise arbitrary ranks will be chosen!
        if compression <= 2: 
            while compression <= compression_factor:         
                '''
                cumulative_rank = ((d**2) *S*T) / (compression_factor*(S/2 + (d**2) + T))
                split_ratio = 0.7 # should be < 0.5
                ranks[0] = np.floor(cumulative_rank * split_ratio).astype(int)
                ranks[1] = np.floor(cumulative_rank - ranks[0]).astype(int)
                '''
                ranks[0] = ranks[0] // 2
                ranks[1] = ranks[1] // 2
                compression = ((d**2) * S * T) / ((S * ranks[0] + ranks[0] * ranks[1] * (d**2) + T * ranks[1]))

        print('compression factor for layer {} : {}'.format(
            weights.shape, compression))
        # Log compression factors and number of weights
        log_compression(weights, compression)

    
    elif flag == 'cpd':
        rank = ranks[0] # it is a single value
        compression = ((d**2)*T*S) / (rank*(S+2*d+T))
        if compression <= 3:
            rank = ((d**2) * S * T) / (compression_factor * (S +2*d+ T))
            ranks[0] = np.floor(rank).astype(int) 

            # recompute new compression ratio 
            compression_factor = ((d**2) * S * T) / (rank * (S +2*d+ T))
            print('compression factor for layer {} : {}'.format(
                weights.shape, compression_factor))
            # Log compression factors and number of weights
            log_compression(weights, compression_factor)
            
        else:
            # Log the standard compression
            log_compression(weights, compression)
            print('compression factor for layer {} : {}'.format(
                weights.shape, compression))
    else:
        #other cases not yet supported
        print('Different decomposition not yet supported!')
        raise(NotImplementedError)

    return ranks


def estimate_ranks(layer):
    """
    Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """
    weights = layer.weight.data.numpy()
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]

    # Check if the VBMF ranks are small enough
    ranks = choose_compression(
        layer, ranks, compression_factor=2, flag='Tucker2')
    return ranks


def cp_ranks(layer):
    weights = layer.weight.data.numpy()
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    #unfold_2 = tl.base.unfold(weights, 3)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    #_, diag_2, _, _ = VBMF.EVBMF(unfold_2)
    print(diag_0.shape[0])
    print(diag_1.shape[1])
    rank=max(diag_0.shape[0], diag_1.shape[1])
    print('VBMF estimated rank:', rank)
    ranks=[rank, rank]
    rank, _=choose_compression(
        layer, ranks, compression_factor=10, flag='cpd')
    return rank

def SVD_weights(weights, t):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
        W: N x M weights matrix
        t: number of singular values to retain
    Returns:
        Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(weights, full_matrices=False)

    U = U[:, :t]
    Sigma = s[:t]
    Vt = V[:t, :]

    L = np.dot(np.diag(Sigma), Vt)
    return U, L 

def FC_SVD_compression(layer):
    """
    Compress a FC layer applying SVD 
    """
    trunc = layer.weight.data.numpy().shape[0]

    weights1, weights2 = SVD_weights(layer.weight, trunc)

    # create SVD FC-layers:
    fc1 = torch.nn.Linear(weights1.shape[0], weights1.shape[1])
    fc2 = torch.nn.Linear(weights2.shape[0], weights2.shape[1])

    fc1.weight.data = torch.from_numpy(np.float32(weights1))
    fc2.weight.data = torch.from_numpy(np.float32(weights2))
    new_layers = [fc1, fc2]
    return nn.Sequantial(*new_layers)

def cp_decomposition_conv_layer(layer, rank, matlab=False):
    """ Gets a conv layer and a target rank, di
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensor.
    print(layer, rank)
    X = layer.weight.data.numpy()
    size = max(X.shape)
    # Using the SVD init gives better results, but stalls for large matrices.

    if matlab: 
        last, first, vertical, horizontal = load_cpd_weights('dumps/TODO.mat')
    
    else:
        if size >= 256:
            print("Init random")
            last, first, vertical, horizontal = parafac(
                X, rank=rank, init='random')
        else:
            last, first, vertical, horizontal = parafac(X, rank=rank, init='svd')

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                             out_channels=first.shape[1],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1],
                                               kernel_size=(
                                                   vertical.shape[0], 1),
                                               stride=layer.stride,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=vertical.shape[1],
                                               bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                                 out_channels=horizontal.shape[1],
                                                 kernel_size=(
                                                     1, horizontal.shape[0]),
                                                 stride=layer.stride,
                                                 padding=(0, layer.padding[0]),
                                                 dilation=layer.dilation,
                                                 groups=horizontal.shape[1],
                                                 bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                             out_channels=last.shape[0],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=True)
    pointwise_r_to_t_layer.bias.data = layer.bias.data

    # Transpose dimensions back to what PyTorch expects
    depthwise_vertical_layer_weights = np.expand_dims(np.expand_dims(
        vertical.transpose(1, 0), axis=1), axis=-1)
    depthwise_horizontal_layer_weights = np.expand_dims(np.expand_dims(
        horizontal.transpose(1, 0), axis=1), axis=1)
    pointwise_s_to_r_layer_weights = np.expand_dims(
        np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1)
    pointwise_r_to_t_layer_weights = np.expand_dims(np.expand_dims(
        last, axis=-1), axis=-1)

    set_layer_weights(depthwise_horizontal_layer,
                      depthwise_horizontal_layer_weights)
    set_layer_weights(depthwise_vertical_layer,
                      depthwise_vertical_layer_weights)
    set_layer_weights(pointwise_s_to_r_layer,
                      pointwise_s_to_r_layer_weights)
    set_layer_weights(pointwise_r_to_t_layer,
                      pointwise_r_to_t_layer_weights)

    '''
    # Fill in the weights of the new layers
    depthwise_horizontal_layer.weight.data = \
        torch.from_numpy(np.float32(depthwise_horizontal_layer_weights))
    depthwise_vertical_layer.weight.data = \
        torch.from_numpy(np.float32(depthwise_vertical_layer_weights))
    pointwise_s_to_r_layer.weight.data = \
        torch.from_numpy(np.float32(pointwise_s_to_r_layer_weights))
    pointwise_r_to_t_layer.weight.data = \
        torch.from_numpy(np.float32(pointwise_r_to_t_layer_weights))
    '''

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer,
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]
    return nn.Sequential(*new_layers)


def cp_decomposition_conv_layer_BN(layer, rank, matlab=False):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensor.
    print(layer, rank)
    X = layer.weight.data.numpy()
    size = max(X.shape)

    if matlab:
        last, first, vertical, horizontal = load_cpd_weights(
            'dumps/TODO.mat')

    else:
        # Using the SVD init gives better results, but stalls for large matrices.
        if size >= 256:
            print("Init random")
            last, first, vertical, horizontal = parafac(
                X, rank=rank, init='random')
        else:
            last, first, vertical, horizontal = parafac(
                X, rank=rank, init='svd')

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                             out_channels=first.shape[1],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1],
                                               kernel_size=(
                                                   vertical.shape[0], 1),
                                               stride=layer.stride,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=vertical.shape[1],
                                               bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                                 out_channels=horizontal.shape[1],
                                                 kernel_size=(
                                                     1, horizontal.shape[0]),
                                                 stride=layer.stride,
                                                 padding=(0, layer.padding[0]),
                                                 dilation=layer.dilation,
                                                 groups=horizontal.shape[1],
                                                 bias=False)

    add_bias = True and layer.bias is not None or False and not layer.bias

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                             out_channels=last.shape[0],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=add_bias)
    if add_bias:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    # Transpose dimensions back to what PyTorch expects
    depthwise_vertical_layer_weights = np.expand_dims(np.expand_dims(
        vertical.transpose(1, 0), axis=1), axis=-1)
    depthwise_horizontal_layer_weights = np.expand_dims(np.expand_dims(
        horizontal.transpose(1, 0), axis=1), axis=1)
    pointwise_s_to_r_layer_weights = np.expand_dims(
        np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1)
    pointwise_r_to_t_layer_weights = np.expand_dims(np.expand_dims(
        last, axis=-1), axis=-1)

    # Fill in the weights of the new layers
    depthwise_horizontal_layer.weight.data = \
        torch.from_numpy(np.float32(depthwise_horizontal_layer_weights))
    depthwise_vertical_layer.weight.data = \
        torch.from_numpy(np.float32(depthwise_vertical_layer_weights))
    pointwise_s_to_r_layer.weight.data = \
        torch.from_numpy(np.float32(pointwise_s_to_r_layer_weights))
    pointwise_r_to_t_layer.weight.data = \
        torch.from_numpy(np.float32(pointwise_r_to_t_layer_weights))

    # create BatchNorm layers wrt to decomposed layers weights
    bn_first = nn.BatchNorm2d(first.shape[1])
    bn_vertical = nn.BatchNorm2d(vertical.shape[1])
    bn_horizontal = nn.BatchNorm2d(horizontal.shape[1])
    bn_last = nn.BatchNorm2d(last.shape[0])

    new_layers = [pointwise_s_to_r_layer, bn_first, depthwise_vertical_layer, bn_vertical,
                  depthwise_horizontal_layer, bn_horizontal,  pointwise_r_to_t_layer,
                  bn_last]
    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    # ranks = [25,40]
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data.numpy(),
                       modes=[0, 1], ranks=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1],
                                  kernel_size=1,
                                  stride=layer.stride,
                                  padding=0,
                                  dilation=layer.dilation,
                                  bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0],
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride,
                                 padding=layer.padding,
                                 dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0],
                                 kernel_size=1,
                                 stride=layer.stride,
                                 padding=0,
                                 dilation=layer.dilation,
                                 bias=True)

    last_layer.bias.data = layer.bias.data

    # Transpose add dimensions to fit into the PyTorch tensors
    first = first.transpose((1, 0))
    first_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(first.copy(), axis=-1), axis=-1)))
    last_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(last.copy(), axis=-1), axis=-1)))
    core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer_BN(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data.numpy(),
                       modes=[0, 1], ranks=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1],
                                  kernel_size=1,
                                  stride=layer.stride,
                                  padding=0,
                                  dilation=layer.dilation,
                                  bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0],
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride,
                                 padding=layer.padding,
                                 dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0],
                                 kernel_size=1,
                                 stride=layer.stride,
                                 padding=0,
                                 dilation=layer.dilation,
                                 bias=True)

    last_layer.bias.data = layer.bias.data

    # Add BatchNorm between decomposed layers
    bn_first = nn.BatchNorm2d(first.shape[1])
    bn_core = nn.BatchNorm2d(core.shape[0])
    bn_last = nn.BatchNorm2d(last.shape[0])

    # Transpose add dimensions to fit into the PyTorch tensors
    first = first.transpose((1, 0))
    first_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(first.copy(), axis=-1), axis=-1)))
    last_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(last.copy(), axis=-1), axis=-1)))
    core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

    new_layers = [first_layer, bn_first,
                  core_layer, bn_core, last_layer, bn_last]
    return nn.Sequential(*new_layers)


def cp_xavier_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensor.
    print(layer, rank)
    weights = layer.weight.data.numpy()


    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=weights.shape[1],
                                             out_channels=rank,
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=rank,
                                               out_channels=rank,
                                               kernel_size=(weights.shape[2], 1),
                                               stride=layer.stride,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=rank,
                                               bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(in_channels=rank,
                                                 out_channels=rank,
                                                 kernel_size=(1, weights.shape[3]),
                                                 stride=layer.stride,
                                                 padding=(0, layer.padding[0]),
                                                 dilation=layer.dilation,
                                                 groups=rank,
                                                 bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=rank,
                                             out_channels=weights.shape[0],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer,
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]

    # Xavier init:
    for l in new_layers:
        xavier_init(l)

    return nn.Sequential(*new_layers)
