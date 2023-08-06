#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os

# Define a function to train the model
def model_train(epoch, n):
    batch_size = 32
    num_classes = 10
    epochs = epoch

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class labels to one-hot encoded vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Build the CNN model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add additional Conv2D and MaxPooling2D layers if n > 1
    if n > 1:
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # Evaluate and save the model
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1] * 100
    model.save("MNIST.h5")
    os.system("mv /MNIST.h5 /mlops-task")
    return accuracy

# Set the number of epochs and layers
no_epoch = 1
no_layer = 1

# Train the model and get the accuracy
accuracy_train_model = model_train(no_epoch, no_layer)

# Write accuracy to a text file
f = open("accuracy.txt", "w+")
f.write(str(accuracy_train_model))
f.close()

# Move the accuracy file to a specific location
os.system("mv /accuracy.txt /mlops-task")
