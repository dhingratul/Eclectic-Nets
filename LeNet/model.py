#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:32:21 2018

@author: dhingratul
"""

# Imports
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
import argparse
import cv2

# Model Definition -- ReLu has been used, that was not in original
# implementation
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

class DNNModel:
    def __init__(self, weights_path = None):
        self.weights_path = weights_path
        self.model = self._build_model()

    def _build_model(self):
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
        if self.weights_path:
            model.load_weights(self.weights_path)
        return model

if __name__ == "__main__":
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to Image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"], 0)
    input_image = image
    X_test = cv2.resize(image, (28,28))
    # Data Normalization
    X_test = X_test / 255.
    # Data reshape for keras format
    X_test = X_test.reshape(-1, 28, 28, 1)
    # Use pre-trained model
    obj = DNNModel(weights_path='lenet.h5')
    obj.model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Predictions
    y_pred = obj.model.predict(X_test)
    prediction = y_pred.argmax(axis=1)
    print("\nPredicted Label", prediction)
    cv2.imshow("input image", input_image)
    cv2.waitKey(0)