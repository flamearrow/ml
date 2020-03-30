import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# create own layer
# __init__: Optionally define sublayers to be used by this layer.
# build: Create the weights of the layer. Add weights with the add_weight method.
# call: Define the forward pass.

class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


model = tf.keras.Sequential([
  MyLayer(10)])

# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)

# ##### Save weights to a TensorFlow Checkpoint file
# model.save_weights('./weights/my_model')
#
# # Restore the model's state,
# # this requires a model with the same architecture.
# model.load_weights('./weights/my_model')


#### Save the model config in a json
import json
import pprint
json_string = model.to_json()

# with open('./model/my_model.json', 'w') as outfile:
#   json.dump(json_string, outfile)

pprint.pprint(json.loads(json_string))

# recreate
# fresh_model = tf.keras.models.model_from_json(json_string)

## save the entire model inlucding definition and weights
model.save('./model/my_model')
