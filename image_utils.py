import matplotlib.pyplot as plt
import tensorflow as tf


def random_invert_img(x, factor=0.5):
    return (255 - x) if tf.random.uniform([]) < factor else x


def get_image_and_label():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    return x_train[0]


# similar function with RandomInvertLayer by using Lambda layer
def create_random_invert_layer(factor=0.5):
    return tf.keras.layers.Lambda(lambda x: random_invert_img(x, factor))


class RandomInvertLayer(tf.keras.layers.Layer):
    def __init__(self, factor=5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x, **kwargs):
        return random_invert_img(x, self.factor)


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


def main():
    img = get_image_and_label()
    # visualize(img, tf.image.flip_left_right(img))
    visualize(img, tf.image.adjust_brightness(img, 0.4))


if __name__ == "__main__":
    main()
