import tensorflow as tf
import os
import shutil
import re
import string
import callback_utils
import io

AUTOTUNE = tf.data.experimental.AUTOTUNE
vocab_size = 1000
embedding_dimension = 16
sequence_length = 100
tensorboard_path = 'tensorboard_logs/moviereview_word_embeddings'

vectors_file_name = 'vecs.tsv'
metadata_file_name = 'meta.tsv'


def get_data():
    # download movie review data
    # also available from tfds: https://www.tensorflow.org/datasets/catalog/imdb_reviews
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    ds_dir = tf.keras.utils.get_file('aclImdb', url, untar=True)
    # for i in os.listdir(ds_dir):
    #     print(i)
    train_dir = os.path.join(ds_dir, 'train')
    test_dir = os.path.join(ds_dir, 'test')
    # removing this so text_dataset_from_dictionary can parse the dirs correctly
    if os.path.exists(os.path.join(train_dir, 'unsup')):
        shutil.rmtree(os.path.join(train_dir, 'unsup'))

    batch_size = 1024
    seed = 123  # shuffle
    train_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, batch_size=batch_size, seed=seed,
                                                                  validation_split=0.2, subset='training')
    val_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, batch_size=batch_size, seed=seed,
                                                                validation_split=0.2, subset='validation')

    # for text_batch, label_batch in train_ds.take(1):
    #   for i in range(5):
    #       print(label_batch[i].numpy(), text_batch.numpy()[i])

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


# input is a data set with text only, no labels, it's used for parameterize TextVectorization layer
def create_vectorizer_layer(text_ds):
    def custom_standerization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation), '')

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standerization,
        max_tokens=vocab_size,
        output_mode='int',
        # input will be cut or truncated
        output_sequence_length=sequence_length
    )
    vectorize_layer.adapt(text_ds)
    return vectorize_layer


def create_embedding_model(text_ds):
    vectorizer_layer = create_vectorizer_layer(text_ds)
    model = tf.keras.Sequential(
        [
            vectorizer_layer,  # output [32, 100] [batch, sentence_length]
            tf.keras.layers.Embedding(vocab_size, embedding_dimension, name="embedding"),
            # output:[32, 100, 5] [batch, sentence_len, dimension]
            # each word of length 100 is mapped to a 5 len embedding
            tf.keras.layers.GlobalAveragePooling1D(),
            # take average on each LINE, [32, 100, 5] becomes [32, 5]
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1),
            # output [1]
        ]
    )
    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def train_embedding_model():
    train_ds, val_ds = get_data()

    # strip label, only training embedding here for TextVectorization
    text_ds = train_ds.map(lambda x, y: x)

    model = create_embedding_model(text_ds)

    tensorboard_callback = callback_utils.create_tensorboard_callback(tensorboard_path)
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=15,
              callbacks=[tensorboard_callback])

    # note can only summary after fit - because the TextVectorization needs actual data to parameterize
    # model.summary()

    # extract the word embeddings
    vectorize_layer = model.get_layer(index=0)
    embedding_layer = model.get_layer(name='embedding')

    vocab = vectorize_layer.get_vocabulary()
    weights = embedding_layer.get_weights()[0]
    print(vocab[:10])
    print(weights.shape)

    out_v = io.open(vectors_file_name, 'w', encoding='utf-8')
    out_m = io.open(metadata_file_name, 'w', encoding='utf-8')

    for num, word in enumerate(vocab):
        if num == 0: continue  # skip padding token from vocab
        vec = weights[num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()


def main():
    train_embedding_model()


if __name__ == "__main__":
    main()
