import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
import numpy as np


if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
    tf.config.set_visible_devices([], 'GPU')

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.first_layer = keras.layers.Flatten(input_shape=(28, 28))
        self.hidden_layer = keras.layers.Dense(128, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.first_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


model = CustomModel()

optimizer = keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, validation_split=0.2, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_images[:1].shape, model.predict(test_images[:1]))

print('\nTest accuracy:', test_acc)
model.save(filepath="models/1", save_format='tf', overwrite=True)
