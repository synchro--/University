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
from keras import optimizers


# others
import scipy.io as sio
import kerassurgeon as ks
from kerassurgeon.operations import insert_layer, delete_layer, replace_layer
import numpy as np
import os

# my files
import cpd
from cpd import *


def build_mnist_cpd_model(model_name, cpd_filename, layer_index=1):
    """Build a custom mnist model from CP decomposition.
       Args:
        model_name: model from which we inherit the architecture and the
                    standard weights.
        cpd_filename: .mat file, contains the decomposed tensors.
       Returns:
        Newly created model
    """

    if not os.path.isfile(model_name):
        print("ERROR: model not found")
        return

    model = load_model(model_name)
    print("base model loaded.")

    # check for specified layer 
    if type(model.layers[layer_index]) is not keras.layers.convolutional.Conv2D:
        print("ERROR: layer " + str(layer_index)
            + "is not a convolutional layer")

    # save first conv layers
    convs = []
    for l in model.layers[:layer_index]:
        convs.append(l)

    # save linear layers after
    linears = []
    up_bound = layer_index + 1
    for l in model.layers[layer_index + 1:]:
        linears.append(l)
    print("saved non separable layers to be used in the final model.")

    print("...Layers below:")
    for l in convs:
        print(l)
    print("...Layers above:")
    for l in linears:
        print(l)

    # load cpd weights and bias and store it
    cpd = load_cpd_weights(cpd_filename)

    # unpack filter weights and bias
    f11, f12, f13, f14, bias = cpd
    print("filters shape:\n")
    print("f11: " + str(f11.shape))
    print("f12: " + str(f12.shape))
    print("f13: " + str(f13.shape))
    print("f14: " + str(f14.shape))


    print("Building Conv2D layers from cpd weights...")
    # create convolutional layers from custom weights
    c21, _ = create_conv2D_from_cpd(f13, 1, 0)
    c22, W22 = create_conv2D_from_cpd(f11, 2, 0)
    c23, W23 = create_conv2D_from_cpd(f12, 3, 0)
    c24, _ = create_conv2D_from_cpd(f14, 4, bias, use_bias=True)
    print("Done.")
    
    
    '''
    print("Building Conv2D layers from cpd weights...")
    # create convolutional layers from custom weights
    c21 = make_conv2D_from_weights(f13, 1, 0)
    c22 = make_conv2D_from_weights(f11, 2, 0)
    c23 = make_conv2D_from_weights(f12, 3, 0)
    c24 = make_conv2D_from_weights(f14, 4, bias, use_bias=True)
    print("Done.")
    '''
    # random test
    '''
    c21 = random_Conv2D(f13, 1, 0)
    c22 = random_Conv2D(f11, 2, 0)
    c23 = random_Conv2D(f12, 3, 0)
    c24 = random_Conv2D(f14, 4, bias, use_bias=True)
    '''
    # change name to avoid conflicts
    c21.name = 'cpd1'
    c22.name = 'cpd2'
    c23.name = 'cpd3'
    c24.name = 'cpd4'

    print("adding previously saved layers...")
    # build up a custom model
    newmodel = Sequential()

    for conv in convs:
        newmodel.add(conv)

    print("first part of the model:")
    newmodel.summary()

    newmodel.add(c21)
    newmodel.add(BatchNormalization())
    newmodel.summary() 
    newmodel.add(c22)
    newmodel.add(BatchNormalization())
    newmodel.add(c23)
    newmodel.add(BatchNormalization())
    newmodel.add(c24)
    newmodel.add(BatchNormalization())

    # newmodel.get_layer('cpd2').input_shape = (None, None, 1)
    # newmodel.get_layer('cpd2').set_weights(W22)
    # newmodel.get_layer('cpd3').input_shape = (None, None, 1)
    # newmodel.get_layer('cpd3').set_weights(W23)

    for l in linears:
        newmodel.add(l)
    print("Done.\n Custom model built succesfully.")
    del model
    newmodel.summary()
    return newmodel


def load_mnist_dataset():
    from keras.datasets import mnist

    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # qui modifica la forma nel caso i canali siano la 2 dimensione o la 4
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255  # normalizing?
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def load_cifar_dataset():
    from keras.datasets import cifar10
    from keras.preprocessing.image import ImageDataGenerator
    num_classes = 10
    data_augmentation = True
    num_predictions = 20

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalizing
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


# Function to fine-tune a custom CPD model on MNIST
def finetune_mnist_cpd(model, lr=1e-3, epochs=5, optimizer="sgd", layer_index=1, freeze_below=True, freeze_above=False, log_file="logs.txt"):
    """Finetune mnist_cpd custom model.
    Args:
        model: the model that has to be fine-tuned.
        lr: Integer, the learning rate. Default to 0.001.
        epochs: number of epochs of training.
        optimizer: type of optimizer used to fine-tune the net.
        layer_index: Integer, index of the convolutional layer that has been substituted.
        freeze_below: Boolean, whether to freeze or not the layers below the 'layer_index'.
        freeze_above: Boolean, whether to freeze or not the layers above the 'layer_index'.
    Returns:
        void.
    """
    batch_size = 128
    epochs = epochs

    x_train, y_train, x_test, y_test = load_mnist_dataset()

    # freeze other layers
    if freeze_below:
        for l in model.layers[:layer_index]:
            print("Freezing layer: " + l.name)
            l.trainable = False

    if freeze_above:
        upper_bound = layer_index + 4
        for l in model.layers[upper_bound:]:
            print("Freezing layer: " + l.name)
            l.trainable = False

    if optimizer == "sgd":
        optim = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(
        optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model compiled. Starting fine-tuning...")
    # train!
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    print("Finetuning complete.\nTesting...")
    # finally, let's test the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Log results to file
    with open("logs.txt", 'a') as out:
        out.write("############################\n")
        out.write("MNIST simple model: Finetuning\n")
        out.write("Freeze layer below: %d " % freeze_below)
        out.write("Freeze layer above: %d\n" % freeze_above)
        out.write("2nd convolutional layer with  CPD\nrank=100 ")
        out.write("lr=%.4f epochs=%d\n" % (lr, epochs))
        out.write("Testing:\n")
        out.write("Test loss: %f\n" % score[0])
        out.write("Test accuracy: %f\n\n" % score[1])


# Fine-tune custom CPD model on CIFAR
def finetune_cifar_cpd(model, lr=1e-3, epochs=5, optimizer="sgd", layer_index=1, freeze_below=True, freeze_above=False, log_file="logs.txt"):
    """Finetune cifar_cpd custom model.
    Args:
        model: the model that has to be fine-tuned.
        lr: Integer, the learning rate. Default to 0.001.
        epochs: number of epochs of training.
        optimizer: type of optimizer used to fine-tune the net.
        layer_index: Integer, index of the convolutional layer that has been substituted.
        freeze_below: Boolean, whether to freeze or not the layers below the 'layer_index'.
        freeze_above: Boolean, whether to freeze or not the layers above the 'layer_index'.
    Returns:
        void.
    """
    batch_size = 32
    epochs = epochs

    x_train, y_train, x_test, y_test = load_cifar_dataset()

    # freeze other layers
    if freeze_below:
        for l in model.layers[:layer_index]:
            print("Freezing layer: " + l.name)
            l.trainable = False

    if freeze_above:
        upper_bound = layer_index + 4
        for l in model.layers[upper_bound:]:
            print("Freezing layer: " + l.name)
            l.trainable = False

    if optimizer == "sgd":
        optim = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(
        optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model compiled. Starting fine-tuning...")
    # train!
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    print("Finetuning complete.\nTesting...")
    # finally, let's test the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Log results to file
    with open(log_file, 'a') as out:
        out.write("############################\n")
        out.write("CIFAR custom CPD model: Finetuning\n")
        out.write("Freeze layer below: %d " % freeze_below)
        out.write("Freeze layer above: %d\n" % freeze_above)
        out.write("4th convolutional layer with  CPD\nrank=128")
        out.write(" lr=%.4f epochs=%d\n" % (lr, epochs))
        out.write("Testing:\n")
        out.write("Test loss: %f\n" % score[0])
        out.write("Test accuracy: %f\n\n" % score[1])


def main():
    '''
    model = load_model('saved_models/mnist.h5')
    folder = "dumps"
    dump_weights(model, folder)


    # Step 1: load CPD tensors, from decomposition computed in Matlab
    mat_contents = sio.loadmat('cpd.mat')['cpd']

    bias = mat_contents['bias'][0][0][0]  # retrieve bias weights

    cpd = mat_contents['weights'][0][0]  # cell of 4 tensors
    f11 = cpd[0][0]
    f12 = cpd[0][1]
    f13 = cpd[0][2]
    f14 = cpd[0][3]


    # the original rank in a CPD decomposition is the size
    # of the 2nd dimension of each rank1 tensor
    rank = f11.shape[1]

    # cpd weights consists of 4 tensors SxR dxR dxR TxR

    # THIS IS ONLY VALID FOR CIFAR10 MODEL
    # Step 2: create thin conv2D layers with using the tensors derived from CPD
    # for 1 layer, 4 convolutions

    in_shape_mnist = (28, 28, 1)
    in_shape_cifar = (32, 32, 3)

    filt = np.random.rand(1, 1, f11.shape[0], f11.shape[1])
    filt[:][:] = f11
    # build weights list
    weights = []
    weights.append(filt)
    weights.append(bias)
    c11 = Conv2D(rank, kernel_size=(1, 1),
                padding='same', activation=None, weights=weights, input_shape=in_shape_cifar)

    # repeat same process with different kernel shape
    c12 = Conv2D(rank, kernel_size=(3, 1), activation=None, weights=f12)
    c13 = Conv2D(rank, kernel_size=(1, 3), activation=None, weights=f13)
    c14 = Conv2D(32, kernel_size=(1, 1), activation='relu', weights=f14)

    # Step3: Network Surgery - substitute original Conv2D layer with the 4 decomposed ones
    # how?

    # Step4: fine-tuning the whole network with very low Learning Rate
    '''

    # 2 layer
    '''
    cpd_11 = mat_contents['f11']
    cpd_12 = mat_contents['f12']
    cpd_13 = mat_contents['f13']
    cpd_14 = mat_contents['f14']
    '''

if __name__ == "__main__":
    main()
