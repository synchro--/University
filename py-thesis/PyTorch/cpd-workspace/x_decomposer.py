######################
## Decomposer Class ##
# For now, it is only a set of functions.
#  TODO:
#  decomposition methods return directly the new decomposed model, i.e. the logic is encapsulated
#  but this would require to do retraining between each decomposition and comunicating that to the user.
#
#   The ranks are estimated with a Python implementation of VBMF
#   https://github.com/CasvandenBogaard/VBMF
#
# Frameworks supported:
# - pytorch
# - keras   (todo!)
# - TF      (todo!)
#
#
#

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF

import collections


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

    ### Public methods
    def estimate_tucker_ranks(self, layer, compression_factor=0):
        """
        Unfold the 2 modes of the specified layer tensor, on which the decomposition
        will be performed, and estimates the ranks of the matrices using VBMF.
        Args:
            layer: the layer that will be decomposed
            compression_factor: preferred compression factor to be enforced over
                                the rank estimation of VBMF, i.e. if VBMF estimated ranks
                                are too high for the desired compression, they will be
                                iteratively divided until they reach the compression rate.
        Returns:
            estimated ranks = [R3, R4]
        """
        weights = layer.weight.data.numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        ranks = [diag_0.shape[0], diag_1.shape[1]]

        if compression_factor:
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
        rank= max(diag_0.shape[0], diag_1.shape[1])
        print('VBMF estimated rank:', rank)
        ranks=[rank, rank]

        # choose desired compression
        if compression_factor:
            rank, _= self._choose_compression(layer, ranks, compression_factor, flag='cpd')
        return rank


    # NB. ideally this should not even have the rank option and be more
    #     intuitive to use.
    def pytorch_cp_layer_decomposition(self, layer, rank, offline=False, filename=''):
        """
            Gets a conv layer and a target rank, and returns a nn.Sequential
            compressed CNN.
            Args:
                layer: the conv layer to decompose
                rank: the rank of the CP-decomposition
                offline: bool, if true the weights will be loaded from the file
                         specified in file name
                filename: string, file from which we have to load the weights.

            Returns:
                The compressed CNN model.
        """
        new_layers = self._cp_decomposition(self, layer, rank, offline, filename)
        return nn.Sequential(*new_layers)


    def pytorch_cp_layer_decomposition_BN(self, layer, rank, offline=False, filename=''):
        """
            Gets a conv layer and a target rank, returns a nn.Sequential compressed CNN,
            with BN units between the compressed layers.
            Args:
                layer: the conv layer to decompose
                rank: the rank of the CP-decomposition
                offline: bool, if true the weights will be loaded from the file
                         specified in file name
                filename: string, file from which we have to load the weights.

            Returns:
                The compressed CNN model.
        """
        first_pointwise, separable_vertical,
        separable_horizontal, last_pointwise = self._cp_decomposition(self, layer, rank, offline, filename)

        # create BatchNorm layers wrt to decomposed layers weights
        bn_first = nn.BatchNorm2d(first.shape[1])
        bn_vertical = nn.BatchNorm2d(vertical.shape[1])
        bn_horizontal = nn.BatchNorm2d(horizontal.shape[1])
        bn_last = nn.BatchNorm2d(last.shape[0])

        new_layers = [first_pointwise, bn_first, separable_vertical, bn_vertical,
                      separable_horizontal, bn_horizontal,  last_pointwise,
                      bn_last]
        return nn.Sequential(*new_layers)

    def FC_SVD_compression(self, layer, trunc=15):
        """
        Compress a FC layer applying SVD
        Returns
            A sequential module containing the 2 compressed layers.
        """
        # trunc = layer.weight.data.numpy().shape[0]
        weights1, weights2 = self._SVD_weights(
            layer.weight.data.cpu().numpy().T, trunc)

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

    def conv1x1_SVD_compression(self, layer, trunc=15):
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
        weights1, weights2 = self._SVD_weights(W.T, trunc)

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


    '''
    # NB. ideally this should not even have the rank option and be more
    #     intuitive to use.
    def pytorch_tucker_decomposition(self, layer, rank, offline=False, filename=''):
        """
            Gets a conv layer and a target rank, and returns a nn.Sequential
            compressed CNN.
            Args:
                layer: the conv layer to decompose
                rank: the rank of the CP-decomposition
                offline: bool, if true the weights will be loaded from the file
                         specified in file name
                filename: string, file from which we have to load the weights.

            Returns:
                The compressed CNN model.
        """
        new_layers = _tucker_decomposition_conv_layer(self, layer)
        return nn.Sequential(*new_layers)


    def pytorch_tucker_decomposition_BN(self, layer, rank, offline=False, filename=''):
        """
        Gets a conv layer and a target rank, returns a nn.Sequential compressed CNN,
        with BN units between the compressed layers.
        Args:
            layer: the conv layer to decompose
            rank: the rank of the CP-decomposition
            offline: bool, if true the weights will be loaded from the file
                        specified in file name
            filename: string, file from which we have to load the weights.

        Returns:
            The compressed CNN model.
    """
    first_pointwise, separable_vertical,
    separable_horizontal, last_pointwise = _tucker_decomposition(la)

    # create BatchNorm layers wrt to decomposed layers weights
    bn_first = nn.BatchNorm2d(first.shape[1])
    bn_vertical = nn.BatchNorm2d(vertical.shape[1])
    bn_horizontal = nn.BatchNorm2d(horizontal.shape[1])
    bn_last = nn.BatchNorm2d(last.shape[0])

    new_layers = [pointwise_s_to_r_layer, bn_first, depthwise_vertical_layer, bn_vertical,
                    depthwise_horizontal_layer, bn_horizontal,  pointwise_r_to_t_layer,
                    bn_last]
    return nn.Sequential(*new_layers)

    # DA rivedere
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


    # DA FARE
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

    '''

    ##################### private utils #######################################
    ###########################################################################

    # Returns tuned ranks according to the desired compression factor
    def _choose_compression(self, layer, ranks, compression_factor=2, flag='Tucker2', framework='pytorch'):
        '''
        Compute tuned ranks according to the desired compression
        factor. Sometimes VBMF returns too large ranks; hence the
        decomposition makes the layer bigger instead of shrinking it.
        This function prevents it.

        N.B. by default, if the compression is higher than 2
        the ranks selected by VBMF will be untouched.


        Args:
            layer: the layer to be compressed
            ranks : estimated ranks
            compression_factor: how much the layer will compressed
            flag: string, choose compression over different decompositions.
                  default is Tucker.
            framework: keywoard specifiying a framework between pytorch, keras and tensorflow.

        Returns:
            the newly estimated rank according to desired compression
        '''

        if framework == 'pytorch':
            # PyTorch format is [OUT, IN, k1, k2]
            weights= layer.weight.data.numpy()
            T = weights.shape[0]
            S = weights.shape[1]
            d = weights.shape[2]

        # if Tucker2 is the selected method, then we compute the correspondent compression factor
        # and then diminish the two ranks R3,R4 iteratively until we reach the desired compression
        if flag == 'Tucker2':
            compression = ((d**2)*S*T) / ((S*ranks[0] + ranks[0]*ranks[1] * (d**2) + T*ranks[1]) )
            ranks[0] = ranks[0] * 3
            print(compression)

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

        # for CPD things are easier, since it depends on a single rank R.
        # hence, we set R get the desired compression and floor it.
        elif flag == 'cpd':
            rank = ranks[0] # it is a single value
            compression = ((d**2)*T*S) / (rank*(S+2*d+T))
            if compression <= compression_factor:
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
            print('[choose compression]: Different decomposition not yet supported!')
            raise(NotImplementedError)

        return ranks


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




    def _cp_decomposition(self, layer, rank, offline=False, filename=''):
        """ Gets a conv layer and a target rank, di
            returns a nn.Sequential object with the decomposition
            Args:
                layer: the conv layer to decompose
                rank: the rank of the CP-decomposition
                offline: bool, if true the weights will be loaded from the file
                         specified in file name
                filename: string, file from which we have to load the weights.

            Returns:
                The compressed 4 layers that substitutes the original one.
        """

        # Perform CP decomposition on the layer weight tensor.
        print('[Decomposer]: computing CP-decomposition of the layer {} with rank {}'.format(layer, rank))
        X = layer.weight.data.numpy()
        size = max(X.shape)

        # THIS SHOULD BE ENHANCED BY USING A SUBPROCESS CALL
        # WHICH CALLS THE MATLAB SCRIPT AND RETRIEVE THE RESULT
        if offline:
            last, first, vertical, horizontal = load_cpd_weights(filename)

        else:
            # SVD init leads to generally better overall compression.
            # However, it can stall for large matrices.
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

        return [first_pointwise, separable_vertical, separable_horizontal, last_pointwise]


 def _tucker_decomposition(self, layer, rank, offline=False, filename=''):
        """ Gets a conv layer and a target rank and
            returns a nn.Sequential object with the decomposition
            Args:
                layer: the conv layer to decompose
                rank: the rank of the CP-decomposition
                offline: bool, if true the weights will be loaded from the file
                         specified in file name
                filename: string, file from which we have to load the weights.

            Returns:
                The compressed 3 layers that substitutes the original one.
        """

        print('[Decomposer]: computing Tucker decomposition of the layer {} with rank {}'.format(layer, rank))

        # THIS SHOULD BE ENHANCED BY USING A SUBPROCESS CALL
        # WHICH CALLS THE MATLAB SCRIPT AND RETRIEVE THE RESULT
        if offline:
            last, first, vertical, horizontal = load_cpd_weights(filename)

        else:
            core, [last, first] = partial_tucker(layer.weight.data.numpy(), modes=[0, 1], ranks=ranks, init='svd')

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
        return new_layers

