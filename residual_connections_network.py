# ML exercise - Oriented Acyclic Graph Neural Networks
# Author: Monika Rosinská
# Date: November 2021

from keras import layers


# TODO: define data x

y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)  # converting input tensor to match y

y = layers.add([y, residual])
