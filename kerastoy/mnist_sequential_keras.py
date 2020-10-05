import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import datetime
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

# (num_samples, 28, 28), (num_samples,)
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist')


# look at one pic
# plt.figure()
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


x_train = x_train / 255.0
x_test = x_test / 255.0

print("x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
print("x_test: {}, y_test: {}".format(x_test.shape, y_test.shape))

# x_train: (60000, 28, 28), y_train: (60000,)
# x_test: (10000, 28, 28), y_test: (10000,)

# Conv2D needs to be NHWC(N, Height, Width, Channel) for CPU
# convert to (600000, 28, 28, 1)
x_train_converted = np.expand_dims(x_train, 3)
x_test_converted = np.expand_dims(x_test, 3)
print("x_train_converted: {}".format(x_train_converted.shape))
print("x_test_converted: {}".format(x_test_converted.shape))



# conv2d model
model = tf.keras.models.Sequential(
  [
    # layers.Flatten(input_shape=(28, 28)),
    layers.Conv2D(64, 3, padding='same', input_shape=(28, 28, 1), data_format='channels_last'),
    # layers.Conv2D(128, 3, data_format='channels_last'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10),
    layers.Softmax()
   ]  # layers
)


# # seq model
# model = tf.keras.models.Sequential(
#   [
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10),
#     layers.Softmax()
#   ]  # layers
# )
#
#
# # print(model.input_shape)
# # (None, 28, 28, 1)
# # print(model.output_shape)
# # (None, 10)
#
#
# loss calcualted on y_true and y_pred
# y_true : [batch_size] - # of examples
# y_pred: [batch_size, num_classes] - # of examples, each has a softmax probability for each category
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# print a loss from un trained model:
# print('untrained loss: {}'.format(loss_fn(y_train[:1], model(x_train_converted[:1]).numpy())))

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train conv2d
model.fit(x_train_converted, y_train, epochs=5, callbacks=[tensorboard_callback])
#
# # train seq
# model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])


