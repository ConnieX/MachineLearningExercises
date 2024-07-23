# ML exercise - Classification to several classes
# Author: Monika Rosinská
# Date: November 2021

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras.utils import to_categorical
from keras.datasets import reuters


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# load data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(path='/kaggle/input/reuters/reuters.npz', num_words=10000)

x_train = vectorize_sequences(train_data)
train_labels = to_categorical(train_labels)
x_test = vectorize_sequences(test_data)
test_labels = to_categorical(test_labels)

x_val = x_train[:1000]
x_train = x_train[1000:]

y_val = train_labels[:1000]
train_labels = train_labels[1000:]

# define model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# show training process
history = model.fit(x_train, train_labels, epochs=9, batch_size=128, validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

# show training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model.evaluate(x_test, test_labels)
