import tensorflow as tf

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import callback_utils
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output

IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128) + (3,)
BATCH_SIZE = 64
BUFFER_SIZE = 1000

OUTPUT_CHANNELS = 3

EPOCHS = 20
VAL_SUBSPLITS = 5

checkpoint_callback_path = 'unet/{epoch:04d}.ckpt'
tensorboard_path = 'logs/unet'
saved_model_path = 'models/unet'


# image: 3 channels, mask: 1 channel
# image nomralized to 0, 1
# masks have value 0, 1, 2
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


# load one point from oxford data:
# FeaturesDict({
#     'file_name': Text(shape=(), dtype=tf.string),
#     'image': Image(shape=(None, None, 3), dtype=tf.uint8),
#     'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=37),
#     'segmentation_mask': Image(shape=(None, None, 1), dtype=tf.uint8),
#     'species': ClassLabel(shape=(), dtype=tf.int64, num_classes=2),
# })
@tf.function  # why do I need this?
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return normalize(input_image, input_mask)


# don't augment
def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)
    return normalize(input_image, input_mask)


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# get train, test set, do some random flip on train set,
# convert values for image to 0, 1, convert values for masks to 0,1,2
def get_data():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    # dataset['train'] and dataset['test'] are dicts with the keys specificed
    # we use map to convert each dict into a tuple of nomalized(image, input_mask)
    # therefore now iterating on train or test would get you (image, input_mask)
    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_ds = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)  # repeat indefinitely
    test_ds = test.batch(BATCH_SIZE)

    return train_ds, test_ds, info


# U-Net implementation:
# The input of our model flows through original MobileNetV2
# then we take the output of some layers from MobileNetV2
#   the layers we took out SHRINKS the image all the way to (4, 4)
# Then we use these intermidiate output as an input some layers of our own model
# Our own model would take the downsampled layers and UPsample them
def create_segmentation_model(output_channels):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False)
    # get some layers from mobilenetv2 as encoder(downsampler)
    # Use the activations of these layers
    # Note only the block_16_project has trainable params, which is fixed
    layer_names = [
        'block_1_expand_relu',
        # block_1_expand_relu (ReLU)      (None, 64, 64, 96)   0           block_1_expand_BN[0][0] - NO training params
        'block_3_expand_relu',
        # block_3_expand_relu (ReLU)      (None, 32, 32, 144)  0           block_3_expand_BN[0][0] - NO training params
        'block_6_expand_relu',
        # block_6_expand_relu (ReLU)      (None, 16, 16, 192)  0           block_6_expand_BN[0][0] - NO training params
        'block_13_expand_relu',
        # block_13_expand_relu (ReLU)     (None, 8, 8, 576)    0           block_13_expand_BN[0][0] - NO training params
        'block_16_project',
        # block_16_project (Conv2D)       (None, 4, 4, 320)    307200      block_16_depthwise_relu[0][0] - have training params
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # create the feature extraction model
    # note output is multiple tensors, i.e the activiations of a bunch of layers
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3),
    ]

    inputs = tf.keras.layers.Input(shape=IMAGE_SHAPE)
    x = inputs

    # creates a Funtional layer that has (128, 128, 3) as input and outputs 5 different tensors
    # these tensors are NOT sequential or connected, they are used separately as input of other layers later
    skips = down_stack(x)

    x = skips[-1]
    # no block_16_project
    skips = reversed(skips[:-1])

    # initial x: (4, 4, 320)
    # skips:        (8 8 576)      (16 16 192)    (32 32 144)       (64 64 96)
    # ups:          (512, 3)       (256, 3)       (128, 3)          (64, 3)
    # up(x):        (8, 8, 512)    (16, 16, 256)  (32, 32, 128)     (64, 64, 64)
    # x = concat:   (8, 8, 1088)   (16, 16, 448)  (32, 32, 272)     (64, 64, 160)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    # now x is (64, 64, 160)

    # Conv2dTranspose INCREASE the channel by strides, it's the reverse of Conv2D
    # (32, 32, 3) -> Conv2DTranspose(256, padding='same', strides=2) -> (64, 64, 256)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same'
    )

    x = last(x)
    # now x is (128, 128, 3)
    return tf.keras.Model(inputs=inputs, outputs=x)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask[0]


def show_predictions(model, ds, num=1):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample_img, sample_mask):
        super(DisplayCallback, self).__init__()
        self.sample_img = sample_img
        self.sample_mask = sample_mask

    def on_train_end(self, logs=None):
        clear_output(wait=True)
        display([self.sample_img, self.sample_mask,
                 create_mask(self.model.predict(self.sample_img[tf.newaxis, ...]))])


def main():
    model = create_segmentation_model(OUTPUT_CHANNELS)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    train_ds, test_ds, info = get_data()

    for sample_image_batch, sample_mask_batch in train_ds.take(1):
        sample_image = sample_image_batch[0]
        sample_mask = sample_mask_batch[0]
    #     display([sample_image, sample_mask,
    #              create_mask(model.predict(sample_image[tf.newaxis, ...]))])
    # model.summary()
    # tf.keras.utils.plot_model(model, to_file='/tmp/model_1.png', show_shapes=True)

    # // unconditionally floor
    steps_per_epoch = info.splits['train'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
    validation_steps = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    model_history = model.fit(train_ds, epochs=EPOCHS,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_data=test_ds,
                              callbacks=[DisplayCallback(sample_image, sample_mask),
                                         callback_utils.create_tensorboard_callback(tensorboard_path),
                                         callback_utils.create_checkpoint_callback(checkpoint_callback_path)])

    model_history.save(saved_model_path)


if __name__ == "__main__":
    main()
