######################
## Decomposer Class ##
# For now, it is only a set of functions.
#  TODO 
#  decomposition methods return directly the new decomposed model, i.e. the logic is encapsulated 
#
#
# Frameworks supported: 
# - pytorch 
# - keras   (todo!)
# - TF      (todo!)

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF

from pytorch_utils import *


class Decomposer(object):

    def __init__(self, model):
        """
        Decomposer module to abstract tensor decomposition methods for 
        Convolutional Neural Networks. 
        Major Features 
        -----------
        - TD block configuration
        - CP decomposition
        - Tucker decomposition
        - arbitrary compression ratio 
        - arbitrary rank 
        - VBMF rank estimation 
        """
        
        self.model = model 
    
    # Public methods
    def estimate_tucker_ranks(self, layer, compression_factor=0):
        """
        Unfold the 2 modes of the specified layer tensor, on which the decomposition 
        will be performed, and estimates the ranks of the matrices using VBMF
        Args: 
            layer: the layer that will be decomposed 
            compression_factor: preferred compression factor to be enforced over
                                the rank estimation of VBMF 
        Returns: 
            estimated ranks = [R3, R4]
        """
        weights = layer.weight.data.numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        ranks = [diag_0.shape[0], diag_1.shape[1]]
        
        
        if choose_compression: 
            # Check if the VBMF ranks are small enough
            ranks = choose_compression(
                layer, ranks, compression_factor=compression_factor, flag='Tucker2')
        
        return ranks


    def estimate_cp_ranks(self, layer, compression_factor=0):
        """
        Unfold the 2 modes of the specified layer tensor, on which the decomposition 
        will be performed, and estimates the ranks of the matrices using VBMF
        Args: 
            layer: the layer that will be decomposed 
            compression_factor: preferred compression factor to be enforced over
                                the rank estimation of VBMF 
        Returns:
            estimated rank R
        """
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
        # choose desired compression 
        if choose_compression:
            rank, _=choose_compression(layer, ranks, 
            compression_factor=compression_factor, flag='cpd')
        return rank
        
    
    # private utils         
    def _SVD_weights(self, weights, t):
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

    def _FC_SVD_compression(self, layer, trunc=15):
        """
        Compress a FC layer applying SVD 
        Returns
            A sequential module containing the 2 compressed layers. 
        """
        # trunc = layer.weight.data.numpy().shape[0]
        weights1, weights2 = self._SVD_weights(layer.weight.data.cpu().numpy().T, trunc)
    
        # create SVD FC-layers:
        fc1 = torch.nn.Linear(weights1.shape[0], weights1.shape[1])
        fc2 = torch.nn.Linear(weights2.shape[0], weights2.shape[1])
        print('created: ')
        print(fc1)
        print(fc2)
    
        fc1.weight.data = torch.from_numpy(np.float32(weights1))
        fc2.weight.data = torch.from_numpy(np.float32(weights2))
        new_layers = [fc1, fc2]
        return nn.Sequential(*new_layers)
        
    
    def _conv1x1_SVD_compression(self, layer, trunc=15):
        """
        Compress a 1x1 conv layer applying SVD 
        Returns: 
            A sequential module containing the 2 compressed layers. 
        """
        # trunc = layer.weight.data.numpy().shape[0]
        W = layer.weight.data.cpu().numpy()
        bias = layer.bias.data
        
        # create the 2 conv layers 
        first = torch.nn.Conv2d(in_channels=W.shape[0],
                                out_channels=trunc,
                                kernel_size=1,
                                stride=layer.stride,
                                padding=0,
                                dilation=layer.dilation,
                                bias=False)
    
        second = torch.nn.Conv2d(in_channels=trunc,
                                out_channels=W.shape[1],
                                kernel_size=1,
                                stride=layer.stride,
                                padding=0,
                                dilation=layer.dilation,
                                bias=True)
        second.bias.data = bias 
    
    
        W = W.T
        W.resize(W.shape[2], W.shape[3])
        weights1, weights2 = SVD_weights(W.T, trunc)
    
        # Transpose dimensions back to what PyTorch expects
        first_weights = np.expand_dims(
            np.expand_dims(weights1.T, axis=-1), axis=-1)
        second_weights = np.expand_dims(np.expand_dims(
            weights2.T, axis=-1), axis=-1)
        
        print(first)
        print(first_weights.shape)
    
        set_layer_weights(first,
                          first_weights)
        set_layer_weights(second,
                          second_weights)
    
    
        new_layers = [first, second]
        return nn.Sequential(*new_layers)
        
        
    def _pytorch_cp_decomposition(self, layer, rank, matlab=False):
        """ Gets a conv layer and a target rank, di
            returns a nn.Sequential object with the decomposition
            Args: 
            
            Returns:
        """
    
        # Perform CP decomposition on the layer weight tensor.
        print(layer, rank)
        X = layer.weight.data.numpy()
        size = max(X.shape)
        
        # THIS SHOULD BE ENHANCED BY USING A SUBPROCESS CALL 
        # WHICH CALLS THE MATLAB SCRIPT AND RETRIEVE THE RESULT 
        if matlab: 
            last, first, vertical, horizontal = load_cpd_weights('dumps/TODO.mat')
        
        else:
            # Using the SVD init gives better results, but stalls for large matrices.
            if size >= 256:
                print("Init random")
                last, first, vertical, horizontal = parafac(
                    X, rank=rank, init='random')
            else:
                last, first, vertical, horizontal = parafac(X, rank=rank, init='svd')
    
        first_pointwise = torch.nn.Conv2d(in_channels=first.shape[0],
                                                 out_channels=first.shape[1],
                                                 kernel_size=1,
                                                 stride=layer.stride,
                                                 padding=0,
                                                 dilation=layer.dilation,
                                                 bias=False)
    
        separable_vertical = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                                   out_channels=vertical.shape[1],
                                                   kernel_size=(
                                                       vertical.shape[0], 1),
                                                   stride=layer.stride,
                                                   padding=(layer.padding[0], 0),
                                                   dilation=layer.dilation,
                                                   groups=vertical.shape[1],
                                                   bias=False)
    
        separable_horizontal = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                                     out_channels=horizontal.shape[1],
                                                     kernel_size=(
                                                         1, horizontal.shape[0]),
                                                     stride=layer.stride,
                                                     padding=(0, layer.padding[0]),
                                                     dilation=layer.dilation,
                                                     groups=horizontal.shape[1],
                                                     bias=False)
    
        last_pointwise = torch.nn.Conv2d(in_channels=last.shape[1],
                                                 out_channels=last.shape[0],
                                                 kernel_size=1,
                                                 stride=layer.stride,
                                                 padding=0,
                                                 dilation=layer.dilation,
                                                 bias=True)
        last_pointwise.bias.data = layer.bias.data
    
        # Transpose dimensions back to what PyTorch expects
        separable_vertical_weights = np.expand_dims(np.expand_dims(
            vertical.transpose(1, 0), axis=1), axis=-1)
        separable_horizontal_weights = np.expand_dims(np.expand_dims(
            horizontal.transpose(1, 0), axis=1), axis=1)
        first_pointwise_weights = np.expand_dims(
            np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1)
        last_pointwise_weights = np.expand_dims(np.expand_dims(
            last, axis=-1), axis=-1)
    
        set_layer_weights(separable_horizontal,
                          separable_horizontal_weights)
        set_layer_weights(separable_vertical,
                          separable_vertical_weights)
        set_layer_weights(first_pointwise,
                          first_pointwise_weights)
        set_layer_weights(last_pointwise,
                          last_pointwise_weights)
    
        new_layers = [first_pointwise, separable_vertical,
                      separable_horizontal, last_pointwise]
        return nn.Sequential(*new_layers)


    def _cp_decomposition_conv_layer_BN(self, layer, rank, matlab=False):
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
    
    
    def _tucker_decomposition_conv_layer(self, layer):
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
    
    
    # Tucker e stabile anche senza BNs. 
    def _tucker_decomposition_conv_layer_BN(self, layer):
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
    
    
    def _tucker_xavier(self, layer):
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
    
        new_layers = [first_layer, core_layer, last_layer]
    
            # Xavier init:
        for l in new_layers:
            xavier_weights2(l)
        
        return nn.Sequential(*new_layers)
    
        
    
    def _cp_xavier_conv_layer(self, layer, rank):
        """ Gets a conv layer and a target rank, 
            returns a nn.Sequential object with the decomposition """
    
        # Perform CP decomposition on the layer weight tensor.
        print(layer, rank)
        weights = layer.weight.data.numpy()
        print(weights.shape[1])
    
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
        print('LOL')
        pointwise_r_to_t_layer.bias.data = layer.bias.data
    
        # create BatchNorm layers wrt to decomposed layers weights
        bn_first = nn.BatchNorm2d(rank)
        bn_vertical = nn.BatchNorm2d(rank)
        bn_horizontal = nn.BatchNorm2d(rank)
        bn_last = nn.BatchNorm2d(weights.shape[0])
    
        new_layers = [pointwise_s_to_r_layer, bn_first, depthwise_vertical_layer, bn_vertical,
                      depthwise_horizontal_layer, bn_horizontal,  pointwise_r_to_t_layer,
                      bn_last]
    
        # Xavier init:
        for l in new_layers:
            xavier_weights2(l)
    
        return nn.Sequential(*new_layers)
