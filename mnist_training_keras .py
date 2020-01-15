# '''Trains a simple convnet on the MNIST dataset.
# Gets to 99.21% test accuracy after 12 epochs
# (there is still a lot of margin for parameter tuning).
# 128 seconds per epoch on a intel i5 CPU.
# '''

# from __future__ import print_function
# import numpy as np
# np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras import backend as K

# ##############################################################################
# ################################# Parameters #################################
# model_save_path="output"
# batch_size = 128
# nb_classes = 10
# nb_epoch = 10

# # input image dimensions
# img_rows, img_cols = 28, 28
# # number of convolutional filters to use
# nb_filters = 32
# # size of pooling area for max pooling
# pool_size = (2, 2)
# # convolution kernel size
# kernel_size = (3, 3)

# ##############################################################################
# ############################### downloading data #############################

# # the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# ##############################################################################
# ############################### LeNet-5 Model ################################

# model = Sequential()

# # first set of CONV => RELU => POOL

# model.add(Convolution2D(20,5,5,border_mode='same',input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# # second set of CONV => RELU => POOL

# model.add(Convolution2D(50, 5, 5, border_mode="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# # set of FC => RELU layers

# model.add(Flatten())
# model.add(Dense(500))
# model.add(Activation("relu"))

# # softmax classifier
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# ##############################################################################
# ########################## Saving model and weights ##########################

# model_json = model.to_json()
# with open("lenet_model1.json", "w") as json_file:
#     json_file.write(model_json)
    
# # Saving weights

# model.save_weights("lenet_weights1.h5")

# tfjs.converters.save_keras_model(model, model_save_path)

# print("Saved model to disk")
# print("training complete!")
 
# organize imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 2
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


hist=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")


