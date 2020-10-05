import tensorflow as tf
import datetime


def create_tensorboard_callback(log_root):
    log_dir = log_root + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def create_checkpoint_callback(ckpt_path):
    return tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1)
