import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
import numpy as np
import optimizer_utils
import plt_utils
import callback_utils

# mobilenet v2 model wih imagenet data
full_classifier_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
headless = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
flower_photos_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
grace_hopper_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
imagenet_labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'

tensorboard_path = 'tensorboard_logs/flower_mobilnetv2'
ckpt_path = 'checkpoints/flower_mobilnetv2/flower-{epoch:04d}.ckpt'
export_path = 'model/flower/ft_from_mobilenetv2'
IMAGE_SHAPE = (224, 224)


# returns an iterator for (x, y)
def load_tensorflow_flower_data():
    flower_photos_root = tf.keras.utils.get_file('flower_photos',
                                                 flower_photos_url,
                                                 untar=True)
    # convert data to 0 - 1
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    return generator.flow_from_directory(str(flower_photos_root), target_size=IMAGE_SHAPE)


# returns the grace hopper image, with value from 0, 1 of shape (1, 244, 244, 3)
def get_grace_hopper_img():
    # download an image to ~/.keras/datasets/image.jpg
    grace_hopper_path = tf.keras.utils.get_file('image.jpg', grace_hopper_url)

    grace_hopper_img = Image.open(grace_hopper_path).resize(IMAGE_SHAPE)
    # grace_hopper_img.show()

    # of shape (224, 224, 3)
    grace_hopper_array = np.array(grace_hopper_img) / 255
    return tf.expand_dims(grace_hopper_array, 0)


def run_tf_hub_model():
    # this is essentially the same with tf.keras.applications.MobileNetV2()
    classifier = tf.keras.Sequential([
        # hub.KerasLayer layer has to accept an 224, 224 input shape
        hub.KerasLayer(full_classifier_model, input_shape=IMAGE_SHAPE + (3,))
    ])

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', imagenet_labels_url)

    labels = np.array(open(labels_path).read().splitlines())

    result = classifier.predict(get_grace_hopper_img())
    result_index = np.argmax(result[0])
    print(labels[result_index])


def create_model_from_feature_vector(num_classes):
    return tf.keras.Sequential([
        # hub.KerasLayer layer has to accept an 224, 224 input shape
        hub.KerasLayer(headless, input_shape=IMAGE_SHAPE + (3,), trainable=False),
        tf.keras.layers.Dense(num_classes)
    ])


def fine_tune_model():
    image_data = load_tensorflow_flower_data()
    model = create_model_from_feature_vector(image_data.num_classes)
    model.compile(optimizer=optimizer_utils.create_rmsprop_optimizer(False),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['acc'])
    steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)  # each epoch use each training data once

    model.fit(image_data,
              epochs=2,
              steps_per_epoch=steps_per_epoch,
              callbacks=[
                  # callback_utils.create_checkpoint_callback(ckpt_path),
                  # callback_utils.create_tensorboard_callback(tensorboard_path),
                  callback_utils.CollectBatchStats()
              ]
              )
    model.save(export_path)


def recreate_from_saved_model():
    image_data = load_tensorflow_flower_data()
    class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])
    model = tf.keras.models.load_model(export_path)
    for image_batch, label_batch in load_tensorflow_flower_data():
        predicted_results = model.predict(image_batch)
        # (32, 5)
        predicted_labels = class_names[np.argmax(predicted_results, axis=-1)]
        actual_labels = class_names[np.argmax(label_batch, axis=-1)]
        plt_utils.draw_30_pic_and_labels(image_batch, predicted_labels, actual_labels=actual_labels)
        break


def main():
    # run_tf_hub_model()
    # fine_tune_model()
    recreate_from_saved_model()


if __name__ == "__main__":
    main()
