# ML exercise - Binary classification
# Author: Monika Rosinská
# Date: November 2021


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path='/kaggle/input/imdb-zip/imdb.npz', num_words=10000)

x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')

x_test =  vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
x_train = x_train[10000:]

y_val = y_train[:10000]
y_train = y_train[10000:]

# define model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# show training process
history = model.fit(x_train, y_train, epochs=2, batch_size=256, validation_data=(x_val, y_val))
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

# show accuracy and validation accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend
plt.show()

# evaluate model
model.evaluate(x_test, y_test)