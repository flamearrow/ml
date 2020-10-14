import callback_utils
import optimizer_utils
import tensorflow as tf
import tensorflow.keras as keras
import argparse

tensorboard_path_exp = 'tensorboard_logs/cifar100_mobilnetv2_keras_exp'
tensorboard_path_cos = 'tensorboard_logs/cifar100_mobilnetv2_keras_cos'
exp_checkpoint_callback_path = 'checkpoints/cifar100_keras_exp/imagenet-{epoch:04d}.ckpt'
cos_checkpoint_callback_path = 'checkpoints/cifar100_keras_cos/imagenet-{epoch:04d}.ckpt'

image_resize = (160, 160)
image_shape = image_resize + (3,)


# create a model with a single dense layer attached at bottom of mobilenetv2, unfreeze mobilenetv2 layers if set
def create_mobilenet_v2_for_cifar100(input_image_shape, fine_tune_layer_starts=-1):
    base_model = keras.applications.MobileNetV2(input_shape=input_image_shape, include_top=False, weights='imagenet')
    # Note: the higher the layer is, the more specific it is to the original training data
    # therefore we want to freeze the lower levels, which learns about generic image features
    if fine_tune_layer_starts >= 0:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_layer_starts]:
            layer.trainable = False
    else:
        base_model.trainable = False
    # print(len(base_model.layers)) # 155 layers in original model
    # now this model has everything up till the softmax layer, the last layer is
    # out_relu (ReLU)                 (None, 1, 1, 1280)   0           Conv_1_bn[0][0]
    # we'll need to convert this layer to a 100 dense output
    # it's tensorflow convention to just call the snake tensor x
    # pooling - convert (x, x, 1280) to (1280) by taking average value on (x, x)
    # for image_size=(30, 30), x = 1
    # for image_size=(160, 160), x = 5
    # we want a slightly bigger number for better results, therefore the images are resized
    global_avg_pool_layer = keras.layers.GlobalAveragePooling2D()
    # add a single 100 length Dense layer as output, no activation, use logits directly
    # Note: 100 here would need to match the total number of possible labels in the models label data
    # if we're doing a binary classification, this number should be 1
    prediction_layer = keras.layers.Dense(100)
    inputs = keras.Input(shape=input_image_shape)
    x = preprocess_inputs(inputs)
    x = base_model(x, training=False)
    x = global_avg_pool_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    return keras.Model(inputs=inputs, outputs=outputs)


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

    # resize the image from (30, 30) to (160, 160) to make mobilenetv2's upper levels wider
    return train_dataset.map(lambda x, y: (tf.image.resize(x, image_resize), y)), test_dataset.map(
        lambda x, y: (tf.image.resize(x, image_resize), y))


def train_mobilenetv2_on_cifar100(fine_tune=False, use_cosine_decay=False):
    # train mobilenetv2 with a simple dense layer for cifar100 data
    # freeze the entire mobilev2 model, only train the last layer
    train_dataset, test_dataset = load_cifar100_data()
    if fine_tune:
        model = create_mobilenet_v2_for_cifar100(image_shape,
                                                 fine_tune_layer_starts=100)
    else:
        model = create_mobilenet_v2_for_cifar100(image_shape)

    model.compile(optimizer=optimizer_utils.create_rmsprop_optimizer(use_cosine_decay=use_cosine_decay),
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
              callbacks=[callback_utils.create_tensorboard_callback(
                  tensorboard_path_cos if use_cosine_decay else tensorboard_path_exp),
                  callback_utils.create_checkpoint_callback(
                      cos_checkpoint_callback_path if use_cosine_decay else exp_checkpoint_callback_path)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-decay", default='exponential')
    args = parser.parse_args()

    if args.decay == 'exponential':
        print('Training with using exponential decay')
        train_mobilenetv2_on_cifar100()
    else:
        print('Training with using cosine decay')
        train_mobilenetv2_on_cifar100(use_cosine_decay=True)


if __name__ == "__main__":
    main()
