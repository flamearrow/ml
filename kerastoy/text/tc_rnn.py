import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import callback_utils
import numpy as np

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIMENSION = 64
tensorboard_path = 'logs/text_classification_lstm'
model_path = 'models/text/text_classification_lstm'


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 64)          523840
# _________________________________________________________________
# bidirectional (Bidirectional (None, None, 128)         66048
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 64)                41216
# _________________________________________________________________
# dense (Dense)                (None, 64)                4160
# _________________________________________________________________
# dropout (Dropout)            (None, 64)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 635,329
# Trainable params: 635,329
# Non-trainable params: 0
# _________________________________________________________________

def get_data():
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE)

    test_dataset = test_dataset.padded_batch(BATCH_SIZE)
    encoder = info.features['text'].encoder

    return train_dataset, test_dataset, encoder


# for a vector, pad zeros so it matches size
def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def create_model(ds_encoder):
    model = tf.keras.Sequential([
        layers.Embedding(ds_encoder.vocab_size, EMBEDDING_DIMENSION),
        # LSTM has (batch, word_count, embedding_dimension] as input and [batch, units] as output
        # using units = 64 here
        # Note LSTM can be configured to output both cell states and hidden states too,
        # this way it can be connected to another LSTM layer
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    return model


def train_model():
    train_ds, test_ds, ds_encoder = get_data()
    model = create_model(ds_encoder)
    # model.summary()
    # tf.keras.utils.plot_model(model, to_file='tc_lstm.png', show_shapes=True)
    model.fit(train_ds, epochs=10, validation_data=test_ds, validation_steps=30,
              callbacks=[callback_utils.create_tensorboard_callback(tensorboard_path)])


def load_saved_model_and_infer():
    model = tf.keras.models.load_model(model_path)
    model.summary()

    _, _, encoder = get_data()

    def sample_predict(predict_text, padding=True):
        encoded_text_vector = encoder.encode(predict_text)
        if padding:
            encoded_text_vector = pad_to_size(encoded_text_vector, EMBEDDING_DIMENSION)
        encoded_text_vector = tf.cast(encoded_text_vector, tf.float32)
        return model.predict(tf.expand_dims(encoded_text_vector, 0))

    sample_positive_predit_text = ('The movie was cool. The animation and the graphics '
                                   'were out of this world. I would recommend this movie. ')

    sample_positive_predit_text = ('The movie was not good. The animation and the graphics '
                                   'were terrible. I would not recommend this movie. ')
    predictions_logits = sample_predict(sample_positive_predit_text)
    # print(predictions)
    # note for binary output, use tf.nn.sigmoid, for multi class, use tf.nn.softmax
    predictions = tf.nn.sigmoid(predictions_logits)
    tf.print(predictions)


def main():
    # train_model()
    load_saved_model_and_infer()


if __name__ == "__main__":
    main()
