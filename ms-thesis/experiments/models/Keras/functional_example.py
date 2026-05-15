# Convolutional Neural Network
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Reshape
from keras.layers.convolutional import Conv2D, Conv3D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
import keras.backend as K 

'''
Functional CNN 
visible = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
hidden1 = Dense(10, activation='relu')(pool2)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='convolutional_neural_network.png')
'''
import numpy 

rank1 = 20
rank2 = 5

x = Input(shape=(32,32,3))
# conv11 = SeparableConv2D(rank1, (3,1))(x)
# conv14 = SeparableConv2D(32, (1,3))(conv11)


conv11 = Conv2D(rank1, kernel_size=1)(x)
# Here we need to expand the input to match Conv3D dimensions 
# sh = conv11.shape + (1,)
#conv11 = Reshape(sh)(conv11)
conv11 = Reshape((32, 32, 1, 20))(conv11)
conv12 = Conv3D(rank1, kernel_size=(3, 1, 1), strides=(1, 1, 1))(conv11)
conv13 = Conv3D(rank1, kernel_size=(1, 3, 1), strides=(1, 1, 1))(conv12)
# Reshape to match Conv2D dimensions
print(conv13.shape)
conv13 = conv13[:,:,:,:,0] 
#conv13 = conv13[:-1]
conv14 = Conv2D(32, kernel_size=1)(conv13)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv14)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
hidden1 = Dense(10, activation='relu')(pool2)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=x, outputs=output)

# summarize layers
print(model.summary())
