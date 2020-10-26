import tensorflow as tf
import os
import unicodedata
import re
import io
from sklearn.model_selection import train_test_split
import time

BATCH_SIZE = 64
ckpt_path = 'checkpoints/text/encoder_decoder_translate'

# reutrns two list of eng and spa words
def get_data(num_examples=100):
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

    # path_to_zip = tf.keras.utils.get_file(
    #     'cmn-eng.zip', origin='http://www.manythings.org/anki/cmn-eng.zip',
    #     extract=True)
    # path_to_file = os.path.dirname(path_to_zip) + "/cmn-eng/cmn.txt"
    lines = io.open(path_to_file, encoding='UTF-8').read().strip().split('\n')
    # each line is converted a pari of [ENGLISH, SPANISH]
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def print_indices(tokenizer, indices):
    for index in indices:
        if index != 0:
            print("%d ----> %s" % (index, tokenizer.index_word[index]))


def tokenize(sentence_list):
    # tokenizer converts a sentece into a vector of tokens of index in the dict
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    # tokenizer has a default dictionary, use this to increase the dictionary size
    tokenizer.fit_on_texts(sentence_list)
    tokens_vector = tokenizer.texts_to_sequences(sentence_list)
    # make sure all tokens have same length
    tokens_vector = tf.keras.preprocessing.sequence.pad_sequences(tokens_vector, padding='post')

    # print some tokens
    # print_indices(tokenizer, tokens_vector[0])

    return tokens_vector, tokenizer


def create_dataset(read_less_lines=True):
    en_list, sp_list = get_data(30000 if read_less_lines else None)

    en_tokens_vector, en_tokenizer = tokenize(en_list)
    sp_tokens_vector, sp_tokenizer = tokenize(sp_list)

    # en_tokens_vector, sp_tokens_vector, en_tokenizer, sp_tokenizer

    # split X, Y into X_train, X_val, Y_train, Y_val, they're all list
    input_train, input_val, output_train, output_val = train_test_split(en_tokens_vector, sp_tokens_vector,
                                                                        test_size=0.2)

    # drop_reminder: if the last batch is not same size with batch size, drop it
    train_ds = tf.data.Dataset.from_tensor_slices((
        input_train, output_train)).shuffle(len(input_train)).batch(BATCH_SIZE,
                                                                    drop_remainder=True)
    test_ds = tf.data.Dataset.from_tensor_slices((input_val, output_val))
    # example_input_batch, example_target_batch = next(iter(train_ds))
    # print(example_input_batch.shape)
    # print(example_target_batch.shape)

    return train_ds, test_ds, en_tokenizer, sp_tokenizer


def unicode_to_ascii(s):
    # NFD
    # Mn: Mark, Nonspacing
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# Takes a sequence and hidden state as input
# output GRU's sequence output[batch, timesteps, units] and hidden states[batch, units]
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoding_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.encoding_units,
                                       # return [batch, time_steps, units
                                       return_sequences=True,
                                       # return states as an additional tensor
                                       return_state=True,
                                       # how to initialize weights
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))


# takes Encoder's output(hidden_state and output)
# outputs attention results[batch, units] and attention weights[batch, timesteps, 1]
# there is no trainable params in attention layer
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_hidden, encoder_output):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(encoder_hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(encoder_output)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# decoder takes a target language input X and encoder output/hidden
# decoder has an attention layer and a embedding-gru-fc layer, does the following:
# 1) run attention on encoder output, get context_vector, attention_weights
# 2) concatenate context_vector and embedding(X)
# 3) run gru on 2), get gru_output and gru_state
# 4) run fc on gru_output to get probability on each vocab
# essentially Decoder takes encoder output and ONE WORD from target language to predict NEXT WORD in target language
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, encoder_hidden, encoder_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(encoder_hidden, encoder_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def train_model():
    # Data
    train_ds, test_ds, en_tokenizer, sp_tokenizer = create_dataset()
    buffer_size = tf.data.experimental.cardinality(train_ds).numpy()
    batch_size = 64
    steps_per_epoch = buffer_size // batch_size
    embedding_dim = 256
    units = 1024
    vocab_input_size = len(en_tokenizer.word_index) + 1
    vocab_output_size = len(sp_tokenizer.word_index) + 1

    example_input_batch, example_target_batch = next(iter(train_ds))

    # Model
    encoder = Encoder(vocab_input_size, embedding_dim, units, batch_size)
    decoder = Decoder(vocab_output_size, embedding_dim, units, batch_size)

    # try a sample run
    # sample_hidden = encoder.initialize_hidden_state()
    # sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))  # (64, 11, 1024)
    # print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))  # (64, 1024)
    #
    # attention_layer = BahdanauAttention(10)
    # attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    # print("Attention result shape: (batch size, units) {}".format(attention_result.shape))  # (64, 1024)
    # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))  # (64, 11, 1)
    #
    # sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
    #                                       sample_hidden, sample_output)
    # print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))  # (64, 9414)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    checkpoint_prefix = os.path.join(ckpt_path, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    # @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            # for once sentence, encoder takes the entire sentence and run once
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            # the start word we input to decoder is <start> token
            dec_input = tf.expand_dims([sp_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            # for each sentence, decoder takes the output of encoder and run once for each word
            # for wordN, input is (encoder_output, decoder_hiddenstate, wordN)
            # output is the probability of worldN+1
            # label is the actual worldN+1
            for t in range(1, targ.shape[1]):
                # targ.shape[0] is 11111111, start from 1
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_ds.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # to evaluate/translate a sentence, we start by building a result[] = <start>
    # run the input sentence on Encoder, get encoder_output, initial_hidden_state
    # then we keep running encoder_output, initial_hidden_state, result[last]
    #  each time decoder predicts the next token from result[last]
    #   resulting in result length increased by one and hidden_state updated
    # keep doing this until the next predicted token is <end>
    def evaluate(sentence):
        sentence = preprocess_sentence(sentence)

        inputs = [en_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=11,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([sp_tokenizer.word_index['<start>']], 0)

        for t in range(16):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += sp_tokenizer.index_word[predicted_id] + ' '

            if sp_tokenizer.index_word[predicted_id] == '<end>':
                return result, sentence

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence


def main():
    train_model()


if __name__ == "__main__":
    main()
