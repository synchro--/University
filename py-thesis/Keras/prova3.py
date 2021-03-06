import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras import backend as K
import numpy as np
import scipy.io as sio
import kerassurgeon as ks

# model = load_model('mnist_custom.h5')

# Dataset
batch_size = 128
num_classes = 10
epochs = 12

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


# Model
rank1 = 4
rank2 = 5
out_dim1 = 32  # T
out_dim2 = 64

model = Sequential()

## DOESN'T WORK 

# 1st convolutional layer
c11 = Conv2D(
    rank1, kernel_size=(1, 1), activation='relu',
            input_shape=input_shape)

c12 = Conv1D(rank1, kernel_size=(3), activation=None)

c13 = Conv1D(rank1, kernel_size=(3), activation=None)

c14 = Conv2D(out_dim1, kernel_size=(1, 1),
             activation='relu')

model.add(c11)
model.add(c12)
model.add(c13)
model.add(c14)

# 2nd convolutional layer
c21 = Conv2D(rank2, kernel_size=(1, 1),
             activation='relu') #input_shape() 

c22 = Conv1D(rank2, kernel_size=(3),
             activation=None)

c23 = Conv1D(rank2, kernel_size=(3),
             activation=None)

c24 = Conv2D(out_dim2, kernel_size=(1, 1),
             activation='relu')

model.add(c21)
model.add(c22)
model.add(c23)
model.add(c24)


# altri layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(
                  lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("mnist_cpd1D.h5")
