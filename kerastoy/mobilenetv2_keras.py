import callback_utils
import tensorflow as tf
import tensorflow.keras as keras
import argparse

tensorboard_path_exp = 'tensorboard_logs/cifar100_mobilnetv2_keras_exp'
tensorboard_path_exp_ft = 'tensorboard_logs/cifar100_mobilnetv2_keras_exp_ft'
tensorboard_path_cos = 'tensorboard_logs/cifar100_mobilnetv2_keras_cos'
tensorboard_path_cos_ft = 'tensorboard_logs/cifar100_mobilnetv2_keras_cos_ft'
exp_checkpoint_callback_path = 'checkpoints/cifar100_keras_exp/imagenet-{epoch:04d}.ckpt'
exp_ft_checkpoint_callback_path = 'checkpoints/cifar100_keras_exp_ft/imagenet-{epoch:04d}.ckpt'
cos_checkpoint_callback_path = 'checkpoints/cifar100_keras_cos_ft/imagenet-{epoch:04d}.ckpt'
cos_ft_checkpoint_callback_path = 'checkpoints/cifar100_keras_cos_ft/imagenet-{epoch:04d}.ckpt'

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


def create_rmsprop_optimizer(use_cosine_decay=False):
    if use_cosine_decay:
        # if start from scratch, use a more aggressive rate
        decay_schedule = keras.experimental.CosineDecayRestarts(
            initial_learning_rate=0.025,
            first_decay_steps=5000,
            alpha=0.001
        )
    else:
        # if fine_tune, use a smaller rate
        decay_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.005,
            decay_steps=5000,
            decay_rate=0.7
        )
    return keras.optimizers.RMSprop(learning_rate=decay_schedule, momentum=0.9, epsilon=1.0)


def train_mobilenetv2_on_cifar100(fine_tune=False, use_cosine_decay=False):
    print("fine_tune: {}, decay: {}".format(fine_tune, "cosine" if use_cosine_decay else "exponential"))
    if use_cosine_decay:
        tensorboard_path = tensorboard_path_cos_ft if fine_tune else tensorboard_path_cos
        ckpt_path = cos_ft_checkpoint_callback_path if fine_tune else cos_checkpoint_callback_path
    else:
        tensorboard_path = tensorboard_path_exp_ft if fine_tune else tensorboard_path_exp
        ckpt_path = exp_ft_checkpoint_callback_path if fine_tune else exp_checkpoint_callback_path

    print("tbpath: {}, ckptpath: {}".format(tensorboard_path, ckpt_path))

    # freeze the entire mobilev2 model, only train the last layer when ft is false
    # otherwise only freeze the first 100 layers of original model(out of 160+ layers)
    train_dataset, test_dataset = load_cifar100_data()
    if fine_tune:
        model = create_mobilenet_v2_for_cifar100(image_shape,
                                                 fine_tune_layer_starts=100)
    else:
        model = create_mobilenet_v2_for_cifar100(image_shape)

    model.compile(optimizer=create_rmsprop_optimizer(use_cosine_decay=use_cosine_decay),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    # model.summary()
    # the two trainable variables are the weights and biases for the trainable dense layer
    # for v in model.trainable_variables:
    #     print(v)
    # x_batch, y_batch = next(iter(test_dataset))
    # output = model(x_batch)
    # print(output)
    # loss0, accuracy0 = model.evaluate(test_dataset)
    # print("initial loss: {:.2f}".format(loss0))
    # print("initial accuracy: {:.2f}".format(accuracy0))

    # if dataset is set, the 2nd param can't be set
    model.fit(train_dataset,
              epochs=1000,
              validation_data=test_dataset,
              callbacks=[callback_utils.create_tensorboard_callback(tensorboard_path),
                         callback_utils.create_checkpoint_callback(ckpt_path)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-decay", default='cosine')
    parser.add_argument("-fine_tune", default='false')
    args = parser.parse_args()

    ft = True if args.fine_tune == 'true' else False
    cos = True if args.decay == 'cosine' else False

    train_mobilenetv2_on_cifar100(fine_tune=ft, use_cosine_decay=cos)


if __name__ == "__main__":
    main()
