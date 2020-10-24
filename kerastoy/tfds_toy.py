import tensorflow as tf

import tensorflow_datasets as tfds


def load_toy():
    # ds, info = tfds.load('mnist', split='train', with_info=True)
    # only works for mnist
    # fig = tfds.show_examples(ds, info)

    ds, info = tfds.load('oxford_iiit_pet:3.*.*', split='train', with_info=True)
    print(info)
    # print(info.features["label"].num_classes)
    print(info.features["label"].names)

    # print(info.features["species"].num_classes)
    print(info.features["species"].names)


def batch_toy():
    # create a ds with numbers
    A = tf.data.Dataset.range(1, 5, output_type=tf.int32)
    # A has data 1,2,3,4

    # change dataset shape
    same_length_ds = A.map(lambda x: tf.fill([2], x))
    # A has data [1 1], [2 2], [3 3], [4 4]
    same_length_batch = same_length_ds.batch(2)
    # one batch has two data
    # will print [1 1] and [2 2]
    for d in same_length_batch.take(1).as_numpy_iterator():
        print(d)

    ragged_ds = A.map(lambda x: tf.fill([x], x))
    # A has data [1], [2 2], [3 3 3], [4 4 4 4]
    ragged_batch = ragged_ds.padded_batch(2)
    # padded_batch will make sure each batch has the same shape by padding them
    # this comes in handy for text input with different lentgh
    # note: different batches will have different shape

    # will print
    # [1 0] and [2 2] for first batch
    # [3 3 3 0] and [4 4 4 4] for second batch
    for d in ragged_batch.take(2).as_numpy_iterator():
        print(d)


def main():
    batch_toy()


if __name__ == "__main__":
    main()
