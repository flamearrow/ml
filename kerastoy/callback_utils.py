import tensorflow as tf
import datetime


def create_tensorboard_callback(log_root):
    log_dir = log_root + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def create_checkpoint_callback(ckpt_path):
    return tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1)


# used to save status of the model at each epoch
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
