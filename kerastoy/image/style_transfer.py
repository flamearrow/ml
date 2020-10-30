import tensorflow as tf
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub


# get a model from tfhub to do style transformation
# https://arxiv.org/abs/1705.06830
def crop_center(image):
    # get a center square of the image of [batch, width, height]
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    # cut a square along the longer edge
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[0] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    return load_image_from_path(tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url), image_size,
                                preserve_aspect_ratio)


def load_image_from_path(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    # read image using matplotlit.pyplot
    img = tf.io.read_file(image_path)
    # if dtype=tf.float32, automatically /255
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.expand_dims(img, 0)

    # img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    # if img.max() > 1.0:
    #     img /= 255
    # img needs to be [1, size, size, 3]
    # if img[0][0][0] == 4:
    #     img = tf.Resi(img, )
    # if len(img.shape) == 3:
    #     # increase the dimension, now it has [229, 229, 3, 3]
    #     img = tf.expand_dims(img, axis=0)
    #     # img = tf.stack([img, img, img], axis=-1)

    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
    return img


def show_n(images):
    original = images[0]
    left_overs = len(images) - 1
    rows = 1 + int(left_overs / 2)

    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 80
    plt.figure(figsize=(w * rows, w))

    plt.subplot(rows, 4, 1)
    plt.imshow(original[0], aspect='equal')
    plt.axis('off')

    for i in range(left_overs):
        plt.subplot(rows, 4, i + 5)
        plt.imshow(images[i + 1][0], aspect='equal')
        plt.axis('off')
    plt.show()
    # n = len(images)
    # # all squares, take the length
    # image_sizes = [image.shape[1] for image in images]
    # # make the canvas big enough
    # w = (image_sizes[0] * 6) // 320
    # plt.figure(figsize=(w * n, w))
    # gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    # for i in range(n):
    #     plt.subplot(gs[i])
    #     # images[size, 1, w, w, 3]
    #     plt.imshow(images[i][0], aspect='equal')
    #     plt.axis('off')
    #     # plt.title(titles[i] if len(titles) > i else '')
    # plt.show()


def get_hub_model():
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    return hub.load(hub_handle)


def run_imgs():
    # content_urls = dict(
    #     sea_turtle='https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg',
    #     tuebingen='https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg',
    #     grace_hopper='https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg',
    # )
    style_urls = dict(
        kanagawa_great_wave='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
        kandinsky_composition_7='https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
        hubble_pillars_of_creation='https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
        van_gogh_starry_night='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
        turner_nantes='https://upload.wikimedia.org/wikipedia/commons/b/b7/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
        munch_scream='https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
        picasso_demoiselles_avignon='https://upload.wikimedia.org/wikipedia/en/4/4c/Les_Demoiselles_d%27Avignon.jpg',
        picasso_violin='https://upload.wikimedia.org/wikipedia/en/3/3c/Pablo_Picasso%2C_1911-12%2C_Violon_%28Violin%29%2C_oil_on_canvas%2C_Kr%C3%B6ller-M%C3%BCller_Museum%2C_Otterlo%2C_Netherlands.jpg',
        picasso_bottle_of_rum='https://upload.wikimedia.org/wikipedia/en/7/7f/Pablo_Picasso%2C_1911%2C_Still_Life_with_a_Bottle_of_Rum%2C_oil_on_canvas%2C_61.3_x_50.5_cm%2C_Metropolitan_Museum_of_Art%2C_New_York.jpg',
        fire='https://upload.wikimedia.org/wikipedia/commons/3/36/Large_bonfire.jpg',
        derkovits_woman_head='https://upload.wikimedia.org/wikipedia/commons/0/0d/Derkovits_Gyula_Woman_head_1922.jpg',
        amadeo_style_life='https://upload.wikimedia.org/wikipedia/commons/8/8e/Untitled_%28Still_life%29_%281913%29_-_Amadeo_Souza-Cardoso_%281887-1918%29_%2817385824283%29.jpg',
        derkovtis_talig='https://upload.wikimedia.org/wikipedia/commons/3/37/Derkovits_Gyula_Talig%C3%A1s_1920.jpg',
        amadeo_cardoso='https://upload.wikimedia.org/wikipedia/commons/7/7d/Amadeo_de_Souza-Cardoso%2C_1915_-_Landscape_with_black_figure.jpg'
    )

    content_image_size = 384
    style_image_size = 256
    # content_images = {k: load_image(v, (content_image_size, content_image_size)) for k, v in content_urls.items()}
    # content_name = 'grace_hopper'  # @param ['sea_turtle', 'tuebingen', 'grace_hopper']
    # style_name = 'munch_scream'  # @param ['kanagawa_great_wave', 'kandinsky_composition_7', 'hubble_pillars_of_creation', 'van_gogh_starry_night', 'turner_nantes', 'munch_scream', 'picasso_demoiselles_avignon', 'picasso_violin', 'picasso_bottle_of_rum', 'fire', 'derkovits_woman_head', 'amadeo_style_life', 'derkovtis_talig', 'amadeo_cardoso']

    style_images = {k: load_image(v, (style_image_size, style_image_size)) for k, v in style_urls.items()}
    style_images = {k: tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME') for k, style_image in
                    style_images.items()}

    hub_module = get_hub_model()

    tora_path = "tora.png"
    tora = load_image_from_path(tora_path, (content_image_size, content_image_size))
    imgs = [tora]

    titles = ['Original content image', 'Style image']
    for name in ['kanagawa_great_wave', 'kandinsky_composition_7', 'hubble_pillars_of_creation',
                 'van_gogh_starry_night', 'turner_nantes', 'munch_scream', 'picasso_demoiselles_avignon',
                 'picasso_violin', 'picasso_bottle_of_rum', 'fire', 'derkovits_woman_head', 'amadeo_style_life',
                 'derkovtis_talig', 'amadeo_cardoso']:
        stylized_image = hub_module(tf.constant(tora),
                                    tf.constant(style_images[name]))[0]
        imgs.append(style_images[name])
        imgs.append(stylized_image)
        titles.append(name)

    # stylized_image = hub_module(tf.constant(content_images[content_name]),
    #                             tf.constant(style_images[style_name]))[0]
    # show_n([content_images[content_name], style_images[style_name], stylized_image],
    #        titles=['Original content image', 'Style image', 'Stylized image'])
    show_n(imgs)


def main():
    run_imgs()


if __name__ == "__main__":
    main()
