import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


# toy feedforward model
class MyFFModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # Define your layers here.
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        return self.dense_2(x)


# toy CNN model
class MyCNNModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyCNNModel, self).__init__(name='my_cnn_model')
        self.num_classes = num_classes
        self.conv2d = layers.Conv2D(64, 3, padding='same', input_shape=(299, 299, 1), data_format='channels_last')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(num_classes)

    # input (none, 299, 299, 3)
    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        # x = self.dense_1(inputs)
        # return self.dense_2(x)
        x = self.conv2d(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

    # need to return a json string
    def get_config(self):
        return {"num_classes": self.num_classes}

    # recover from config, this is essentially a factory function
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config['num_classes']) if 'num_classes' in config else cls()


def create_sequential_toy_model():
    return keras.Sequential(
        [
            layers.Conv2D(64, 3, padding='same', input_shape=(28, 28, 1), data_format='channels_last'),
            # layers.Conv2D(128, 3, data_format='channels_last'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10),
            layers.Softmax()
        ]
    )


def create_functional_model():
    inputs = keras.Input((32,))
    outputs = keras.layers.Dense(10)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def main():
    model = MyFFModel(num_classes=10)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)


if __name__ == "__main__":
    main()
