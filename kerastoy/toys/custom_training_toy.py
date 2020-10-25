import tensorflow as tf
import matplotlib.pyplot as plt


# use GradientTape to do custom training on a model instead of using fit

# steps:
# define model: i.e NN and parameters to train - can write own layer/model and define params within
# define loss: i.e how to compute loss between y_true and model(input)
# train: for each step
# use with tf.GradientTape() as tape: to wrap model.call() and loss.call()
# outside the with block, use grads = tape.gradient(loss_value, params) to get the gradient in the previous run
# send gradient and params to optimizer to update the params: optimizer.apply_gradients(zip(grads, params))

def create_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                                   input_shape=(None, None, 1)),
            tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)
        ]
    )


def get_data():
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
         tf.cast(mnist_labels, tf.int64)))
    dataset = dataset.shuffle(1000).batch(32)
    return dataset


def train_step(images, labels, model, loss, loss_history, optimizer):
    # use tape to record the computations during one step
    # when a batch of image flows through the model and gets their logits
    with tf.GradientTape() as tape:
        logits = model(images)
        tf.debugging.assert_equal(logits.shape, (32, 10))
        loss_value = loss(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    # tape can return the gradients from loss with regard to variables
    gradients = tape.gradient(loss_value, model.trainable_variables)
    # 'gradient descend' by applying gradients to optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_model_custom(epoch=3):
    model = create_model()
    ds = get_data()
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_history = []
    # instead of using fit, write how to train on each step
    for epoch in range(epoch):
        for (batch, (images, labels)) in enumerate(ds):
            print('Batch {} of Epoch {}'.format(batch, epoch))
            train_step(images, labels, model, loss_object, loss_history, optimizer)
        print('Epoch {} finished'.format(epoch))

    plt.plot(loss_history)
    plt.xlabel('Batch #')
    plt.ylabel('Loss [entropy]')
    plt.show()


# GradientTape is used to record calculation on trainable tf.Variables and then calculates
# the gradient for some params with regards to the result
def gradient_tape_toy():
    # w = tf.Variable([1.0])
    w = tf.Variable(tf.random.normal((3, 2)), name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    t = tf.zeros((2,), dtype=tf.float32)
    x = [[1., 2., 3.]]

    with tf.GradientTape() as tape:
        # the calculation happens inside the graph to be recorded by tape
        # in real model training, this involves sending a batch of Xs into model to get Y
        # then use loss function to calculate the loss
        # y = w * w
        # @: matmul
        # can optionally watch tensors
        # tape.watch(t)
        y = x @ w + b + t
        loss = tf.reduce_mean(y ** 2)

    print("watched vars: ")
    # Note t is not watched here is it's a Tensor not a Variable
    for var in tape.watched_variables():
        print(var)
    print("==============")
    # calculate gradient of y with regards to w, note here w could be multiple variables,
    # in real model, w is the total trainable params of a model
    # based on what param is passed, it would return corresponding gradient(same shape of the variable)
    _ = tape.gradient(loss, (w, b))

    # can also calculate gradient with regards to intermediate output inside the recording
    # note tape.gradient() can only be called once
    # _ = tape.gradient(loss, y)


def main():
    gradient_tape_toy()
    # train_model_custom()


if __name__ == "__main__":
    main()
