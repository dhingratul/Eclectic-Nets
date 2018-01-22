#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:48:53 2018

@author: dhingratul
"""
from __future__ import print_function
from LeNet import leNet
import cv2
import argparse

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
model = leNet(weights_path='lenet.h5')
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Predictions
y_pred = model.predict(X_test)
prediction = y_pred.argmax(axis=1)
print("\nPredicted Label", prediction)
cv2.imshow("input image", input_image)
cv2.waitKey(0)
