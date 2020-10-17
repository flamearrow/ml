import argparse

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import callback_utils
import optimizer_utils

# Note: model will overfit(high accuracy on training set but low on val set) when the model is too complicated for the
# date, e.g model has way too many params compared to not enough training data, in this case if we turn on fine tune,
# the model probably will overfit(See 99% accuracy on train but only 60% on val)
# some running results

# finetune from layer 100 in mobilenet -saved under blahblah_ft
# ingraph: - data loaded from tensorflow_datasets
#  cosine_finetue_in_graph: 200 epoch to converge - 99% train and 60% eval
#  exp_finetue_in_graph: 50 epoch to converge - 96% train and 63% eval
#  cosine_no_finetune_in_graph: 250 epoch - 42% train and 46% eval
#  exp_no_finetune_in_graph: 40%train and 47% eval
# outgraph: - data loaded from tf.keras.datasets
#  cosine_finetune_out_graph: 25 epoch - 80% train and 58% eval
#  exp_finetune_out_graph: probably similar, e.g 96-60
#  cosine_no_finetune_out_graph: probably similar, e.g 50-60
#  cos_no_finetune_out_graph: 60 58 running


# finetine from layer x in mobilenet -saved under blahblah_ft_start_layer
#  130: cosine_finetue_in_graph: 90 60
#  150: cosine_finetue_in_graph: 90 60
#  152: cosine_finetue_out_graph: 85 65
#  153: exp_finetue_out_graph: 58 61
#  153: cosine_finetue_out_graph: 60 60


# no fintune, the result dangles around 42%
# with finetune, got overfit


FT_start_layer = 153
tensorboard_path_exp = 'tensorboard_logs/cifar100_mobilnetv2_keras_exp'
tensorboard_path_exp_ft = 'tensorboard_logs/cifar100_mobilnetv2_keras_exp_ft_' + str(FT_start_layer) + '_'
tensorboard_path_cos = 'tensorboard_logs/cifar100_mobilnetv2_keras_cos'
tensorboard_path_cos_ft = 'tensorboard_logs/cifar100_mobilnetv2_keras_cos_ft_' + str(FT_start_layer) + '_'
exp_checkpoint_callback_path = 'checkpoints/cifar100_keras_exp/imagenet-{epoch:04d}.ckpt'
exp_ft_checkpoint_callback_path = 'checkpoints/cifar100_keras_exp_ft_' + str(
    FT_start_layer) + '_/imagenet-{epoch:04d}.ckpt'
cos_checkpoint_callback_path = 'checkpoints/cifar100_keras_cos/imagenet-{epoch:04d}.ckpt'
cos_ft_checkpoint_callback_path = 'checkpoints/cifar100_keras_cos_ft_' + str(
    FT_start_layer) + '_/imagenet-{epoch:04d}.ckpt'

exp_model_path = 'model/cifar100/from_mobilnetv2_exp'
exp_ft_model_path = 'model/cifar100/from_mobilnetv2_exp_ft_' + str(FT_start_layer) + '_'
cos_model_path = 'model/cifar100/from_mobilnetv2_cos'
cos_ft_model_path = 'model/cifar100/from_mobilnetv2_cos_ft_' + str(FT_start_layer) + '_'

total_epochs = 250
batch_size = 32
image_width = 160
image_resize = (image_width, image_width)
image_shape = image_resize + (3,)
AUTOTUNE = tf.data.experimental.AUTOTUNE


# create a model with a single dense layer attached at bottom of mobilenetv2, unfreeze mobilenetv2 layers if set
def create_mobilenet_v2_for_cifar100(input_image_shape, fine_tune_layer_starts=-1, in_graph_preprocess=True):
    # 155 layers total
    base_model = keras.applications.MobileNetV2(input_shape=input_image_shape, include_top=False, weights='imagenet')
    # Note: the higher the layer is, the more specific it is to the original training data
    # therefore we want to freeze the lower levels, which learns about generic image features
    if fine_tune_layer_starts >= 0:
        print("fine tuning from layer {} ".format(fine_tune_layer_starts))
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_layer_starts]:
            layer.trainable = False
    else:
        base_model.trainable = False

    if in_graph_preprocess:
        # print(len(base_model.layers)) # 155 layers in original model
        # now this model has everything up till the softmax layer, the last layer is
        # out_relu (ReLU)                 (None, 1, 1, 1280)   0           Conv_1_bn[0][0]
        # we'll need to convert this layer to a 100 dense output
        # it's tensorflow convention to just call the snake tensor x
        # pooling - convert (x, x, 1280) to (1280) by taking average value on (x, x)
        # for image_size=(30, 30), x = 1
        # for image_size=(160, 160), x = 5
        # we want a slightly bigger number for better results, therefore the images are resized
        # add a single 100 length Dense layer as output, no activation, use logits directly
        # Note: 100 here would need to match the total number of possible labels in the models label data
        # if we're doing a binary classification, this number should be 1
        inputs = keras.Input(shape=input_image_shape)
        x = resize_and_rescale()(inputs)
        x = in_graph_data_augmentation()(x)
        x = base_model(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(100)(x)
        return keras.Model(inputs=inputs, outputs=outputs)
    else:
        # out graph, don't introduce preprocessing
        return keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(100),
        ])


def in_graph_data_augmentation():
    return keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])


def resize_and_rescale():
    return keras.Sequential([
        keras.layers.experimental.preprocessing.Resizing(image_width, image_width),
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])


def out_graph_data_augment(image, label):
    print("apply out_graph_data_augment")
    image = resize_and_rescale()(image)
    # Add 6 pixels of padding
    image = tf.image.resize_with_crop_or_pad(image, image_width + 6, image_width + 6)
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[image_width, image_width, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    image = tf.image.flip_left_right(image)
    image = tf.clip_by_value(image, 0, 1)
    return image, label


# use keras.datasets.cifar100 to load data, apply data augmentation and resizing outside tfgraph
# note when apply out of graph augmentation, do more steps than in graph augmentation
def load_cifar100_data_out_graph_preprocessing():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    # print("x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
    # print("x_test: {}, y_test: {}".format(x_test.shape, y_test.shape))
    # x_train: (50000, 32, 32, 3), y_train: (50000, 1)
    # x_test: (10000, 32, 32, 3), y_test: (10000, 1)

    # Rescaling: mobilenetv2 needs pixels values in -1, 1
    # the converted values are not drawable
    # x_train_rescaled = x_train / 127.5 - 1
    # x_test_rescaled = x_test / 127.5 - 1

    # # convert numpy arrays into tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # apply full augmentation to training set, only apply resize and rescale to test set
    train_dataset = train_dataset.shuffle(1000).map(out_graph_data_augment, num_parallel_calls=AUTOTUNE).batch(
        batch_size).prefetch(AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (resize_and_rescale()(x), y), num_parallel_calls=AUTOTUNE).batch(
        batch_size).prefetch(AUTOTUNE)

    # # prefetch for better perf... do I need this as it's converted in memory?
    # train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    #
    # # resize the image from (30, 30) to (160, 160) to make mobilenetv2's upper levels wider
    # train_dataset = train_dataset.map(lambda x, y: (resize_and_rescale()(x), y), num_parallel_calls=AUTOTUNE)
    # test_dataset = test_dataset.map(lambda x, y: (resize_and_rescale()(x), y), num_parallel_calls=AUTOTUNE)
    # # augment the data by introducing some randomness
    #
    # data_aug = tf.keras.Sequential([
    #     keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #     keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # ])
    #
    # data_aug = out_graph_data_augment()
    # train_dataset = train_dataset.map(lambda x, y: (data_aug(x, training=True), y),
    #                                   num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset, test_dataset


# use tfds.load() to load cifar100 data, don't apply data augmentation and resizing for datasets, let the tfgraph do it
def load_cifar100_in_graph_preprocessing():
    # 80, 10, 10
    (train_ds, val_ds, test_ds), metadata = tfds.load('cifar100',
                                                      split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                      with_info=True,
                                                      as_supervised=True)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    train_ds = train_ds.shuffle(1000)
    # no need to shuffle val and test
    # val_ds = val_ds.shuffle(1000)
    # test_ds = test_ds.shuffle(1000)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # classes = metadata.features['label'].num_classes
    # # function to get label from index
    # get_label_fn = metadata.features['label'].int2str
    # image, label = next(iter(val_ds))
    # _ = plt.imshow(image)
    # _ = plt.title(get_label_fn(label))
    # plt.show()

    # image, label = next(iter(train_ds))
    # expanded_img = tf.expand_dims(image, 0)
    # plt.figure(figsize=(10, 10))
    # aug = data_augmentation()
    # for i in range(9):
    #     augmented_image = aug(expanded_img)
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(augmented_image[0])
    #     plt.axis("off")
    # plt.show()
    return train_ds, val_ds, test_ds


def train_mobilenetv2_on_cifar100(fine_tune=False, use_cosine_decay=False, in_graph_preprocess=True):
    print("fine_tune: {}, decay: {}, in_graph_preprocess: {}".format(fine_tune,
                                                                     "cosine" if use_cosine_decay else "exponential",
                                                                     in_graph_preprocess))
    if use_cosine_decay:
        tensorboard_path = tensorboard_path_cos_ft if fine_tune else tensorboard_path_cos
        ckpt_path = cos_ft_checkpoint_callback_path if fine_tune else cos_checkpoint_callback_path
        model_path = cos_model_path if fine_tune else cos_ft_model_path
    else:
        tensorboard_path = tensorboard_path_exp_ft if fine_tune else tensorboard_path_exp
        ckpt_path = exp_ft_checkpoint_callback_path if fine_tune else exp_checkpoint_callback_path
        model_path = exp_model_path if fine_tune else exp_ft_model_path

    print("tbpath: {}, ckptpath: {}, modelpath:{}".format(tensorboard_path, ckpt_path, model_path))

    model = create_mobilenet_v2_for_cifar100(image_shape,
                                             fine_tune_layer_starts=FT_start_layer if fine_tune else -1,
                                             in_graph_preprocess=in_graph_preprocess)

    model.compile(optimizer=optimizer_utils.create_rmsprop_optimizer(use_cosine_decay=use_cosine_decay),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    # model.summary()

    train_dataset, validation_dataset, test_dataset = \
        load_cifar100_in_graph_preprocessing() if in_graph_preprocess else load_cifar100_data_out_graph_preprocessing()
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
              epochs=total_epochs,
              validation_data=validation_dataset,
              callbacks=[callback_utils.create_tensorboard_callback(tensorboard_path),
                         callback_utils.create_checkpoint_callback(ckpt_path)])

    # loss0, accuracy0 = model.evaluate(test_dataset)
    # print("initial loss: {:.2f}".format(loss0))
    model.save(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-decay", default='cosine')
    parser.add_argument("-fine_tune", default='false')
    parser.add_argument("-in_graph_preprocess", default='true')
    args = parser.parse_args()

    ft = True if args.fine_tune == 'true' else False
    cos = True if args.decay == 'cosine' else False
    in_graph_preprocess = True if args.in_graph_preprocess == 'true' else False

    train_mobilenetv2_on_cifar100(fine_tune=ft, use_cosine_decay=cos, in_graph_preprocess=in_graph_preprocess)


if __name__ == "__main__":
    main()
