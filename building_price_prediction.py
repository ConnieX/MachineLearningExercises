# ML exercise - Regression
# Author: Monika Rosinská
# Date: November 2021

import numpy as np
from keras import models, layers
from keras.datasets import boston_housing


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# load data
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data(path='/kaggle/input/housing/boston_housing.npz')

# normilise values into <-1; 1> interval
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# define constants
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

# process data and fit them into model
for i in range(k):
    print("Proccesing fold #", i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[ : i * num_val_samples], train_data[(i + 1) * num_val_samples : ]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_labels[ : i * num_val_samples], train_labels[(i + 1) * num_val_samples : ]],
        axis=0)
    
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# define constants for validation data
k = 4
num_val_samples = len(train_data) // k
num_epochs = 200
all_mae_histories = []

# process data and fit them into model and check its performance on validation data
for i in range(k):
    print("Proccesing fold #", i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[ : i * num_val_samples], train_data[(i + 1) * num_val_samples : ]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_labels[ : i * num_val_samples], train_labels[(i + 1) * num_val_samples : ]],
        axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt

# show training process
plt.plot(range(1, len(average_mae_history) +1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# show training process with smoothened curve
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.clf()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# fit model and investigate the scores
model = build_model()
model.fit(train_data, train_labels, epochs = 50, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)
print(f"{test_mae_score}")
