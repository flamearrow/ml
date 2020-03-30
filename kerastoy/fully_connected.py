import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras import layers

# creates a tensor
inputs = tf.keras.Input(shape=(32,))  # Returns an input placeholder

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10)(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)