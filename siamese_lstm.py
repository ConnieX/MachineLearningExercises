# ML exercise - Siamese (Shared) LSTM; sharing one layer and its weights in two parts of the network
# Author: Monika Rosinská
# Date: November 2021

from keras import layers, Input, applications
from keras.models import Model

lstm = layers.LSTM(32)

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

# common classificator
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_output, right_output], predictions)
model.fit([left_data, right_data], targets)

# DEPTH VERSION
xception_base = applications.Xception(weights=None, include_top=False)

left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

left_features = xception_base(left_input)
right_features = xception_base(right_input)

merged_features = layers.concatenate([left_features, right_features], axis=-1)
