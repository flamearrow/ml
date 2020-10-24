import argparse
import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#
# tf.debugging.set_log_device_placement(True)

# Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
# print(c)


def test_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-blah", default='heyo')
    args = parser.parse_args()
    for _ in range(100000):
        print(args.blah)


def main():
    print(tf.random.normal([32, 10, 8]))


if __name__ == "__main__":
    main()
