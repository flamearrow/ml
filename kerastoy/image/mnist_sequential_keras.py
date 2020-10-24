import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import callback_utils

# ckpt names with epoch number concatenated
seq_checkpoint_callback_path = 'mnist_keras/seq-{epoch:04d}.ckpt'
cnn_checkpoint_callback_path = 'mnist_keras/cnn-{epoch:04d}.ckpt'

seq_checkpoint_direct_save_path = 'mnist_keras/seq-saved.ckpt'
cnn_checkpoint_direct_save_path = 'mnist_keras/cnn-saved.ckpt'

tensorboard_path = 'logs/mnist_keras'

# add .h5 to save to hdf5 format seq_savedmodel_path = 'mnist_keras/seq_model.h5'

seq_savedmodel_path = '../mnist_keras/seq_model'
cnn_savedmodel_path = 'mnist_keras/cnn_model'


def create_sequential_model():
    model = tf.keras.models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10),
            layers.Softmax()
        ]  # layers
    )
    # loss calcualted on y_true and y_pred
    # y_true : [batch_size] - # of examples
    # y_pred: [batch_size, num_classes] - # of examples, each has a softmax probability for each category
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print(model.input_shape)
    # (None, 28, 28, 1)
    print(model.output_shape)
    # (None, 10)
    model.summary()
    return model


def create_cnn_model():
    model = tf.keras.models.Sequential(
        [
            layers.Conv2D(64, 3, padding='same', input_shape=(28, 28, 1), data_format='channels_last'),
            # layers.Conv2D(128, 3, data_format='channels_last'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10),
            layers.Softmax()
        ]  # layers
    )
    # loss calculated on y_true and y_pred
    # y_true : [batch_size] - # of examples
    # y_pred: [batch_size, num_classes] - # of examples, each has a softmax probability for each category
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.summary()
    return model


def get_datas():
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
    return x_train, y_train, x_test, y_test, x_train_converted, x_test_converted


def train_model_and_save_ckpt(save_weights_callback=False, seq_model=True):
    x_train, y_train, x_test, y_test, x_train_converted, x_test_converted = get_datas()

    # model = create_cnn_model();

    model = create_sequential_model()

    # train conv2d
    # model.fit(x_train_converted, y_train, epochs=5, callbacks=[create_tensorboard_callback('logs/fit')])

    callbacks = [callback_utils.create_tensorboard_callback(tensorboard_path)]

    if save_weights_callback:
        callbacks.append(
            callback_utils.create_checkpoint_callback(
                seq_checkpoint_callback_path if seq_model else cnn_checkpoint_callback_path))

    if seq_model:
        model.fit(
            x_train,
            y_train,
            epochs=5,
            # need validation_data during fit in order for the checkpoint to decide 'best' one
            validation_data=(x_test, y_test),
            callbacks=callbacks)
    else:
        model.fit(
            y_train,
            x_test_converted,
            epochs=5,
            # need validation_data during fit in order for the checkpoint to decide 'best' one
            validation_data=(x_test, y_test),
            callbacks=callbacks)
    # manually save weights
    if not save_weights_callback:
        model.save_weights(seq_checkpoint_direct_save_path if seq_model else cnn_checkpoint_direct_save_path)


def train_model_and_save_savedmodel(seq_model=True):
    x_train, y_train, x_test, y_test, x_train_converted, x_test_converted = get_datas()

    # model = create_cnn_model();

    model = create_sequential_model()

    # train conv2d
    # model.fit(x_train_converted, y_train, epochs=5, callbacks=[create_tensorboard_callback('logs/fit')])

    if seq_model:
        model.fit(
            x_train,
            y_train,
            epochs=5,
            # need validation_data during fit in order for the checkpoint to decide 'best' one
            validation_data=(x_test, y_test))
    else:
        model.fit(
            x_train_converted,
            x_test_converted,
            epochs=5,
            # need validation_data during fit in order for the checkpoint to decide 'best' one
            validation_data=(x_test, y_test))

    model.save(seq_savedmodel_path if seq_model else cnn_savedmodel_path)


# create the model, don't train but load checkpoints trained from before
def recreate_model_from_ckpt(load_callback_ckpt=True, seq_model=True):
    x_train, y_train, x_test, y_test, x_train_converted, x_test_converted = get_datas()

    if seq_model:
        model = create_sequential_model()
        loss, acc = model.evaluate(x_test, y_test, verbose=1)
        print("Untrained seq model, loss: {}, accuracy: {:5.2f}%".format(loss, 100 * acc))

        model.load_weights(tf.train.latest_checkpoint(
            os.path.dirname(seq_checkpoint_callback_path)) if load_callback_ckpt else seq_checkpoint_direct_save_path)
        loss, acc = model.evaluate(x_test, y_test, verbose=1)
        print("Pretrained seq model, loss: {}, accuracy: {:5.2f}%".format(loss, 100 * acc))
    else:
        model = create_cnn_model()
        loss, acc = model.evaluate(x_test_converted, y_test, verbose=1)
        print("Untrained cnn model, loss: {}, accuracy: {:5.2f}%".format(loss, 100 * acc))

        model.load_weights(tf.train.latest_checkpoint(
            os.path.dirname(cnn_checkpoint_callback_path)) if load_callback_ckpt else cnn_checkpoint_direct_save_path)
        loss, acc = model.evaluate(x_test_converted, y_test, verbose=1)
        print("Pretrained cnn model, loss: {}, accuracy: {:5.2f}%".format(loss, 100 * acc))


# create the model, don't train but load checkpoints trained from before
def recreate_model_from_saved_model(seq_model=True):
    x_train, y_train, x_test, y_test, x_train_converted, x_test_converted = get_datas()

    if seq_model:
        model = tf.keras.models.load_model(seq_savedmodel_path)
        model.summary()
        loss, acc = model.evaluate(x_test, y_test, verbose=1)
        print("Pretrained seq model recreated from savedmodel, loss: {}, accuracy: {:5.2f}%".format(loss, 100 * acc))
    else:
        model = tf.keras.models.load_model(cnn_savedmodel_path)
        model.summary()
        loss, acc = model.evaluate(x_test_converted, y_test, verbose=1)
        print("Pretrained cnn model recreated from savedmodel, loss: {}, accuracy: {:5.2f}%".format(loss, 100 * acc))


def main():
    # train_model(save_weights_callback=True)
    # recreate_model_from_ckpt(load_callback_ckpt=False)
    # train_model_and_save_savedmodel()
    recreate_model_from_saved_model()


if __name__ == "__main__":
    main()
