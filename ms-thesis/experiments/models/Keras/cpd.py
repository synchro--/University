# loading custom weights defined with CPD
# for matlab

# Keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

# others
import scipy.io as sio
import kerassurgeon as ks
from kerassurgeon.operations import insert_layer, delete_layer, replace_layer
import numpy as np
import os

'''
 ## questo va bene se le dimensioni non cambiassero ... in CPD serve istanziare un layer nuovo con pesi diversi
 ## anzi 4 layer nuovi LOL

 # ritorna 2 vettori uno per i pesi e uno per i bias
 model.layers[7].get_weights()
 weight_shape = model.layers[7].get_weights()[0].shape
 bias_shape = model.layers[7].get_weights()[1].shape

 # costruiamo i due vettori da inserire nei layer
 custom_weights = []
 custom_weights.append(np.random.rand(weight_shape[0], weight_shape[1]))
 custom_weights.append(np.random.rand(bias_shape[0], bias_shape[1]))
 model.layers[7].set_weights(custom_weights)

 # si può aggiungere un layer con pesi custom, in questa maniera:
 model.add(Dense(1, activation='softmax', weights = ins))

 # per inserirli nel mezzo usiamo il pacchetto surgeon e si vola

 Steps:
 creare il layer dai pesi ottenuti con CPD
 inserirlo con surgeon nel posto giusto (why doesn't work??)
 si ottiene un keras.training.model
 si fa:

 newmodel = Sequential()
 for l in model.layers:
     newmodel.add(l)
'''


def dump_weights(model, folder_name="dumps"):
    '''
    Dump weights for all Conv2D layers and saves it as .mat files
    TODO: Add check if file exists
    '''
    save_dir = os.path.join(os.getcwd(), folder_name)
    # create dir if not exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    allweights = []  # general list
    for l in model.layers:
        if(type(l) == keras.layers.convolutional.Conv2D):
            allweights.append(l.get_weights())

    for idx, weights in enumerate(allweights):
        name = os.path.join(save_dir, "conv" + str(
            idx) + ".mat")  # conv1.mat, conv2.mat, ...
        sio.savemat(name,  {'weights': weights})


def create_conv2D_from_cpd(filter_weights, layer_order, bias, use_bias=False, activation=None, is_first=False, input_shape=None):
    """Create a Conv2D keras layer from custom weights, filter_weights.
    Args:
        filter_weigths: Numpy array, custom weights of the Conv2D filter
        bias: Numpy array, The bias vector of the layer. Only used if use_bias is True.
        use_bias: Boolean, whether the layer uses the bias vector 'bias'.
        activation: Boolean, whether an activation layer is applied after convolution.
        layer_order: Integer, specify the order of the layer in the 4 of the decomposition, starting from 1.
        is_first: Boolean, whether the layer is the first of the model.
        input_shape: Tuple, if the layer is the first of the model, specifies the expected shape.

    Returns:
        A Conv2D layer with the specified custom weights.
    """
    # the original rank in a CPD decomposition is the size
    # of the 2nd dimension of each rank1 tensor
    rank = filter_weights.shape[1]
    delta = filter_weights.shape[0]

    if layer_order == 1:
        # weights = np.random.rand(1, 1, delta, rank)
        weights = filter_weights.reshape(1, 1, delta, rank)
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if use_bias:
            W.append(bias)

        if is_first:
            C = Conv2D(rank, kernel_size=(1, 1), activation=activation,
                       weights=W, use_bias=use_bias, input_shape=input_shape)
        else:
            C = Conv2D(rank, kernel_size=(1, 1),
                       activation=activation, weights=W, use_bias=use_bias, padding="same")

    elif layer_order == 2:
        # weights = np.zeros(delta, 1, rank, rank)
        # this is how they do in the paper:
        # weights = np.zeros(delta, 1, 1, rank)
        weights = filter_weights.reshape(delta, 1, 1, rank)

        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if use_bias:
            W.append(bias)

        # but we need to know what is the output shape of the layer before :D
        # so the whole pipeline must be modified
        # WE NEED A FUNCTION TO BUILD ONLY ONE LAYER AT A TIME
        C = Conv2D(
            rank, kernel_size=(delta, 1), input_shape=(None, None, None, 1),
                   activation=activation, use_bias=use_bias)

    elif layer_order == 3:
        weights = filter_weights.reshape(1, delta, 1, rank)

        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if use_bias:
            W.append(bias)

        C = Conv2D(rank, kernel_size=(1, delta),
                   activation=activation, use_bias=use_bias, input_shape=(None, None, None, 1))

    else:
        weights = np.zeros((1, 1, rank, delta))
        weights[:, :] = filter_weights.transpose()
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        W.append(bias)
        output_dim = delta
        C = Conv2D(output_dim, kernel_size=(1, 1),
                   activation=activation, weights=W, use_bias=True)

    return C, W 


def make_conv2D_from_weights(filter_weights, layer_order, bias, use_bias=False, activation=None, is_first=False, input_shape=None):
    """Create a Conv2D keras layer from custom weights, filter_weights.
    Args:
        filter_weigths: Numpy array, custom weights of the Conv2D filter
        bias: Numpy array, The bias vector of the layer. Only used if use_bias is True.
        use_bias: Boolean, whether the layer uses the bias vector 'bias'.
        activation: Boolean, whether an activation layer is applied after convolution.
        layer_order: Integer, specify the order of the layer in the 4 of the decomposition, starting from 1.
        is_first: Boolean, whether the layer is the first of the model.
        input_shape: Tuple, if the layer is the first of the model, specifies the expected shape.

    Returns:
        A Conv2D layer with the specified custom weights.
    """
    # the original rank in a CPD decomposition is the size
    # of the 2nd dimension of each rank1 tensor
    rank = filter_weights.shape[1]
    delta = filter_weights.shape[0]

    if layer_order == 1:
        weights = np.zeros((1, 1, delta, rank))
        weights[:, :] = filter_weights
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if use_bias:
            W.append(bias)

        if is_first:
            C = Conv2D(rank, kernel_size=(1, 1), activation=activation,
                       weights=W, use_bias=use_bias, input_shape=input_shape)
        else:
            C = Conv2D(rank, kernel_size=(1, 1),
                       activation=activation, weights=W, use_bias=use_bias)

    elif layer_order == 2:
        weights = np.zeros((delta, 1, rank, rank))
        # this is how they do in the paper:
        # weights = np.zeros(delta, 1, 1, rank)
        # weights = filter_weights.reshape(delta, 1, 1,rank)
        # forse c'è da aggiungere un input_shape = dentro al conv

        # set the weights at the proper slices ... (check if correct!)
        for idx in range(0, rank):
            weights[:, 0, :, idx] = filter_weights

        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if use_bias:
            W.append(bias)

      #  C = Conv2D(rank, kernel_size=(delta, 1),
      #             activation=activation, weights=W, use_bias=use_bias)

        C = Conv2D(rank, kernel_size=(delta, 1), input_shape=(None, None, None, 1), activation=activation, weights=W, use_bias=use_bias)

    elif layer_order == 3:
        weights = np.zeros((1, delta, rank, rank))
        # weights = np.zeros(1, delta, 1, rank)
        # weights = filter_weights.reshape(1, delta, 1,rank)

        # set the weights at the proper slices ... (check if correct!)
        for idx in range(0, rank):
            weights[0, :, :, idx] = filter_weights

        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if use_bias:
            W.append(bias)

       # C = Conv2D(rank, kernel_size=(1, delta),
       #            activation=activation, weights=W, use_bias=use_bias)

        C = Conv2D(rank, kernel_size=(1, delta), input_shape=(
        None, None, None, 1), activation=activation, weights=W, use_bias=use_bias)

    else:
        weights = np.zeros((1, 1, rank, delta))
        weights[:, :] = filter_weights.transpose()
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        W.append(bias)
        output_dim = delta
        C = Conv2D(output_dim, kernel_size=(1, 1),
                   activation=activation, weights=W, use_bias=True)

    return C


def load_cpd_weights(filename):
        import scipy.io as sio
        import os

        if not os.path.isfile(filename):
            print("ERROR: .mat file not found")
            return

        # load struct 'cpd_s' from file
        mat_contents = sio.loadmat(filename)['cpd_s']

        bias = mat_contents['bias'][0][0][0]  # retrieve bias weights

        cpd = mat_contents['weights'][0][0]  # cell of 4 tensors
        f11 = cpd[0][0]
        f12 = cpd[0][1]
        f13 = cpd[0][2]
        f14 = cpd[0][3]
        print('Loaded cpd weights succesfully.')

        return f11, f12, f13, f14, bias


# dummy function to create conv layer with random weithgs
def random_Conv2D(filter_weights, layer_order, bias, use_bias=False, activation=None, is_first=False, input_shape=None):

    # the original rank in a CPD decomposition is the size
    # of the 2nd dimension of each rank1 tensor
    rank = filter_weights.shape[1]
    delta = filter_weights.shape[0]

    if layer_order == 1:
        weights = np.random.rand(1, 1, delta, rank)
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        if is_first:
            C = Conv2D(rank, kernel_size=(1, 1), activation=activation,
                       weights=W, input_shape=input_shape)
        else:
            C = Conv2D(rank, kernel_size=(1, 1),
                       activation=activation, weights=W, use_bias=False)

    elif layer_order == 2:
        weights = np.random.rand(delta, 1, rank, rank)
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        C = Conv2D(rank, kernel_size=(delta, 1),
                   activation=activation, weights=W, use_bias=False)

    elif layer_order == 3:
        weights = np.random.rand(1, delta, rank, rank)
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        C = Conv2D(rank, kernel_size=(1, delta),
                   activation=activation, weights=W, use_bias=False)

    else:
        weights = np.random.rand(1, 1, rank, delta)
        bias = np.random.rand(delta,)
        # create list for Conv2D set_weights API
        W = []
        W.append(weights)
        W.append(bias)
        output_dim = delta
        C = Conv2D(output_dim, kernel_size=(1, 1),
                   activation=activation, weights=W, use_bias=True)

    return C


def conv2d_bn_cpd(x,
                  filters,
                  rank,
                  num_row,
                  num_col,
                  padding='valid',
                  strides=(1, 1),
                  name=None):
    """Utility function to apply 4-decomposed convs+BN

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`, it will be the output of the last conv layer.
        rank: rank approx. of the tensor. See CPD.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization` 4 times according to the CPD architecture.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = 'cpd_bn_1'
        conv_name = 'cpd1'
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    cnames = (conv_name, "cpd2", "cpd3", "cpd4")
    bnnames = (bn_name, "bn_cpd2", "bn_cpd3")

    x = Conv2D(
        filters, kernel_size=(1, 1),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=(cnames[0]))(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bnnames[0])(x)

    x = Conv2D(
        rank, kernel_size=(num_row, 1),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=(cnames[1]))(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bnnames[1])(x)

    x = Conv2D(
        rank, kernel_size=(1, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=(cnames[2]))(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bnnames[2])(x)

    x = Conv2D(
        filters, kernel_size=(1, 1),
        strides=strides,
        padding=padding,
        use_bias=True,
        name=(cnames[3]))(x)

    x = Activation('relu', name="cpd_act")(x)
    return x
