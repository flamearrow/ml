import tensorflow as tf
import callback_utils
import tensorflow.keras as keras
import plt_utils

tensorboard_path = 'logs/cifar100_mobilnetv2_keras'
seq_checkpoint_callback_path = 'cifar100_keras/imagenet-{epoch:04d}.ckpt'
image_shape = (32, 32, 3)


# create a model with a single dense layer attached at bottom of mobilenetv2, unfreeze mobilenetv2 layers if set
def createMobilenetV2ForCifar100(image_shape, fine_tune_layer_starts=-1):
    base_model = keras.applications.MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    # Note: the higher the layer is, the more specific it is to the original training data
    # therefore we want to freeze the lower levels, which learns about generic image features
    if fine_tune_layer_starts >= 0:
        base_model.trainable = True
        for layer in base_model[:fine_tune_layer_starts]:
            layer.trainable = False
    else:
        base_model.trainable = False
    # print(len(base_model.layers)) # 155 layers in original model
    # now this model has everything up till the softmax layer, the last layer is
    # out_relu (ReLU)                 (None, 1, 1, 1280)   0           Conv_1_bn[0][0]
    # we'll need to convert this layer to a 100 dense output
    # it's tensorflow convention to just call the snake tensor x
    # pooling - convert (x, x, 1280) to (1280) by taking average value on (x, x)
    global_avg_pool_layer = keras.layers.GlobalAveragePooling2D()
    # add a single 100 length Dense layer as output, no activation, use logits directly
    # Note: 100 here would need to match the total number of possible labels in the models label data
    # if we're doing a binary classification, this number should be 1
    prediction_layer = keras.layers.Dense(100)
    inputs = keras.Input(shape=image_shape)
    x = preprocess_inputs(inputs)
    x = base_model(x, training=False)
    x = global_avg_pool_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# This doesn't seem to wark
# class MobileNetV2ForCifar100(keras.Model):
#     def __init__(self):
#         super(MobileNetV2ForCifar100, self).__init__()
#         # cifar100 is 32 by 32 by 3
#         self.base_model = keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False)
#         self.pool_layer = keras.layers.GlobalAveragePooling2D()
#         self.logits_layer = keras.layers.Dense(100)
#
#     def call(self, inputs, training=None, mask=None):
#         self.base_model(inputs)
#         x = self.base_model.output
#         x = self.pool_layer(x)
#         x = self.logits_layer(x)
#         return x


# flip, rotate andd set scale from (0-255) to (-1, 1) for input images
def preprocess_inputs(input_layer):
    return keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
        keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)  # rescale value to -1, 1
    ])(input_layer)


def load_cifar100_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    # print("x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
    # print("x_test: {}, y_test: {}".format(x_test.shape, y_test.shape))
    # x_train: (50000, 32, 32, 3), y_train: (50000, 1)
    # x_test: (10000, 32, 32, 3), y_test: (10000, 1)

    # Rescaling: mobilenetv2 needs pixels values in -1, 1
    # the converted values are not drawable
    # x_train_rescaled = x_train / 127.5 - 1
    # x_test_rescaled = x_test / 127.5 - 1

    BATCH_SIZE = 32

    # # convert numpy arrays into tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    # shuffle the entire set
    train_dataset = train_dataset.shuffle(50000)
    test_dataset = test_dataset.shuffle(10000)

    # prefetch for better perf... do I need this as it's converted in memory?
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def create_rmsprop_optimizer(fine_tune=True):
    if fine_tune:
        # if fine_tune, use a smaller rate
        decay_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.005,
            decay_steps=5000,
            decay_rate=0.7
        )
    else:
        # if start from scratch, use a more aggressive rate
        decay_schedule = keras.experimental.CosineDecayRestarts(
            initial_learning_rate=0.025,
            first_decay_steps=5000,
            alpha=0.001
        )
    return keras.optimizers.RMSprop(learning_rate=decay_schedule, momentum=0.9, epsilon=1.0)


def train_mobilenetv2_on_cifar100(fine_tune=False):
    # train mobilenetv2 with a simple dense layer for cifar100 data
    # freeze the entire mobilev2 model, only train the last layer
    train_dataset, test_dataset = load_cifar100_data()
    if fine_tune:
        model = createMobilenetV2ForCifar100(image_shape,
                                             fine_tune_layer_starts=100)
        train_optimizer = create_rmsprop_optimizer(fine_tune=fine_tune)
    else:
        model = createMobilenetV2ForCifar100(image_shape)
        train_optimizer = create_rmsprop_optimizer()

    model.compile(optimizer=train_optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    # model.summary()

    # the two trainable variables are the weights and biases for the trainable dense layer
    # for v in model.trainable_variables:
    #     print(v)

    # x_batch, y_batch = next(iter(test_dataset))
    # output = model(x_batch)
    # print(output)

    loss0, accuracy0 = model.evaluate(test_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    # if dataset is set, the 2nd param can't be set
    # this is going to suck as
    # a) there are 100 categories compared to cat/dog category
    # b) we only added one dense layer and froze the entire mobilenet, there's not too much to play around
    model.fit(train_dataset,
              epochs=1000,
              validation_data=test_dataset,
              callbacks=[callback_utils.create_tensorboard_callback(tensorboard_path),
                         callback_utils.create_checkpoint_callback(seq_checkpoint_callback_path)])


def main():
    train_mobilenetv2_on_cifar100()


if __name__ == "__main__":
    main()
