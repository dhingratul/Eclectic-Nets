#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:32:21 2018

@author: dhingratul
"""

# Imports
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential
from keras.layers.core import Activation


# Model Definition -- ReLu has been used, that was not in original
# implementation
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
def leNet(weights_path=None):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=3, strides=1, padding='same',
                     input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation("relu"))
    model.add(Conv2D(16, kernel_size=5, strides=1, padding='valid'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation("relu"))
    model.add(Dense(84))
    model.add(Dense(10, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model
