# ML exercise - Multiple Inputs Network
# Author: Monika Rosinská
# Date: November 2021

import numpy as np
from keras.models import Model
from keras import layers, Input


# define constants
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# define model
text_input = Input(shape=(None,), dtype='int32', name='text') # input is text of variable length
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input) # inputs stored into vector sequence of size 64
encoded_text = layers.LSTM(32)(embedded_text) # coding vectors into one vector

question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)  #connect question and text

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)  # add clasificator

# define model
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# random data
num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

# fit model
model.fit([text, question], answers, epochs=10, batch_size=128)
#model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
