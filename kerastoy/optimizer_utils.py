import tensorflow.keras as keras


def create_rmsprop_optimizer(use_cosine_decay=False):
    if use_cosine_decay:
        # if start from scratch, use a more aggressive rate
        decay_schedule = keras.experimental.CosineDecayRestarts(
            initial_learning_rate=0.025,
            first_decay_steps=5000,
            alpha=0.001
        )
    else:
        decay_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.005,
            decay_steps=5000,
            decay_rate=0.7
        )
    return keras.optimizers.RMSprop(learning_rate=decay_schedule, momentum=0.9, epsilon=1.0)


def create_adam_optimizer():
    return keras.optimizers.Adam()
