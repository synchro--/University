
'''Train a simple deep CNN with Custom CPD layer on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger

from cpd import conv2d_bn_cpd
import os
import time
import numpy as np


batch_size = 32
num_classes = 10
epochs = 15
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

rank1 = 20
rank2 = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

## 1
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
'''
model.add(Conv2D(rank1, kernel_size=(1, 1),
          padding='same', input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(3, 1), input_shape=(None, None, None, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(1, 3), input_shape=(None, None, None, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(1, 1)))
model.add(Activation('relu'))
'''
input_shape3d = np.zeros((1,32,32,3))
input_shape3d[0] = x_train.shape[1:]

## New Model with Conv3D 
model.add(Conv3D(rank1, kernel_size=(1, 1, 3),
                 padding='same', input_shape=input_shape3d.shape))
model.add(BatchNormalization())
model.add(Conv3D(rank1, kernel_size=(3, 1, 1), input_shape=(None, None, None, None, 1)))
model.add(BatchNormalization())
model.add(Conv3D(rank1, kernel_size=(1, 3, 1), input_shape=(None, None, None, None, 1)))
model.add(BatchNormalization())
model.add(Conv3D(32, kernel_size=(1, 1, rank1)))
model.add(Activation('relu'))

## 2
# model.add(Conv2D(32, (3, 3)))
model.add(Conv2D(rank1, kernel_size=(1, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(3, 1), input_shape=(None, None, None, 1))) # input_shape=(30, 30, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(1, 3), input_shape=(None, None, None,1))) # input_shape=(28, 30, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(1, 1)))
model.add(Activation('relu'))

## 3
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

## 4
# model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(rank1, kernel_size=(1, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(3, 1), input_shape=(14, 14, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(1, 3), input_shape=(12, 14, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(1, 1)))
model.add(Activation('relu'))

## 5
# model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(rank2, kernel_size=(1, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank2, kernel_size=(3, 1), input_shape=(12, 12, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank2, kernel_size=(1, 3), input_shape=(10, 12, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(1, 1)))
model.add(Activation('relu'))

## 6
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
'''
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
'''

## 7
model.add(Conv2D(rank1, kernel_size=(1, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(5, 1), input_shape=(5, 5, 1)))
model.add(BatchNormalization())
model.add(Conv2D(rank1, kernel_size=(1, 5), input_shape=(1, 5, 1)))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size=(1, 1)))

## 8 Classifier
model.add(Conv2D(num_classes, kernel_size=(1, 1), activation=None))
model.add(Flatten())
model.add(Activation('softmax'))

# load model weights, if saved

# model.load_weights("weights.best.hdf5")
# print("loadad weights!")

model.summary()

print("weights ")
print(model.layers[2].get_weights()[0].shape)
print(model.layers[4].get_weights()[0].shape)

print(model.layers[8].get_weights()[0].shape)
print(model.layers[10].get_weights()[0].shape)


# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_log = CSVLogger('log.csv', separator=",", append=False)
callbacks_list = [checkpoint, csv_log]

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list,
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,
            # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,
            # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,
            # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # timing training
    train_t0 = time.time()

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list,
                        workers=4)

    train_t1 = time.time()
    total_time = train_t1 - train_t0

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
print('\nTraining total time: ====> ', total_time)

# Score trained model.
test_t0 = time.time()
scores = model.evaluate(x_test, y_test, verbose=1)
test_t1 = time.time()
total_time = test_t1 - test_t0
print('\nTesting total time: ====> ', total_time)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
