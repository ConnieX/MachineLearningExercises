# ML exercise - Data Augmentation (Convolutional Neural Network)
# Author: Monika Rosinská
# Date: November 2021

import matplotlib.pyplot as plt
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator


# load data
train_dir = '/kaggle/input/dog-vs-cats/train'
test_dir = '/kaggle/input/dog-vs-cats/test'
val_dir = '/kaggle/input/dog-vs-cats/validation'
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=16, class_mode='binary')
val_generator = test_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=16, class_mode='binary')

# define model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# show training process
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=val_generator, validation_steps=50)
model.save('cats_vs_dogs_2.keras')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validaton acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validaton loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
