# ML exercise - practical functions like early stopping, model checkpoints, reduction of plateau, creations of callback
# Author: Monika Rosinská
# Date: November 2021

import keras
import numpy as np
from keras import model, callbacks, layers

# TODO: load data

# early stopping and model checkpoint
callbacks_list=[
    callbacks.EarlyStopping(monitor='acc', patience='1'),  # patience stop training, if correctnes stop improving for MORE than x (there x = 1) epoch, that means x+1 epochs
    callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True)  # saves the model after each epoch, if its better than the saved one
]
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=[x_val, y_val])

# reduction of plateau - lowers learning rate if valiadation loss stops improving; allows to get away from local minimum
callbacks_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)] # learning rate multiplies by factor after calling (after patience+1 epochs without metrics improvement)
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=[x_val, y_val])

# creation of new callback
# can be called: on_epoch_begin, on_epoch_end, on_batch_begin, on_batch_end, on_train_begin, on_train_end
# example of callback taht saves activation of all layers at the end of each epoch (values for first sample of validation dataset)
class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layers.outputs = [layer.output for layer in model.layers]
        self.activation_model = keras.models.Model(model.input, layer_outputs)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activation_model.predict(validation_sample)
        
        f.open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
