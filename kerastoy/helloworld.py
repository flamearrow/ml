import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras import layers

model = tf.keras.Sequential([
  # Adds a densely-connected layer with 64 units to the model:
  layers.Dense(64, activation='relu', input_shape=(32,)),
  # Add another:
  layers.Dense(64, activation='relu'),
  # Add an output layer with 10 output units:
  layers.Dense(10)])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#####train with different data format
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

# train from numpy data
# model.fit(data, labels, epochs=10, batch_size=32)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

# convert numpy data to tf.data and train from tf.data
model.fit(dataset, epochs=10,
          validation_data=val_dataset)

############# evalue and predict with different data format
# With Numpy arrays
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

# With a Dataset
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.evaluate(dataset)
