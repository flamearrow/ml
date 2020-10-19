import tensorflow as tf

import tensorflow_datasets as tfds


def main():
    # ds, info = tfds.load('mnist', split='train', with_info=True)
    # only works for mnist
    # fig = tfds.show_examples(ds, info)

    ds, info = tfds.load('oxford_iiit_pet:3.*.*', split='train', with_info=True)
    print(info)
    # print(info.features["label"].num_classes)
    print(info.features["label"].names)

    # print(info.features["species"].num_classes)
    print(info.features["species"].names)


if __name__ == "__main__":
    main()
