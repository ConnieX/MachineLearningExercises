# ML exercise - Simple Neural Network
# Author: Monika Rosinská
# Date: November 2021

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, models, layers
from keras.utils import to_categorical


# load data
train_images = ((np.load('/kaggle/input/mnist-dataset/x_train1.npz')).reshape(60000, 28*28)).astype('float32') / 255
train_labels = to_categorical(np.load('/kaggle/input/mnist-dataset/y_train1.npy'))
test_images = ((np.load('/kaggle/input/mnist-dataset/x_test1.npy')).reshape(10000, 28*28)).astype('float32') / 255
test_labels = to_categorical(np.load('/kaggle/input/mnist-dataset/y_test1.npy'))

# define model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc', test_acc)

# show example image
train_img = np.load('/kaggle/input/mnist-dataset/x_train1.npy')
plt.imshow(train_img[1], cmap=plt.cm.binary)
plt.show()
