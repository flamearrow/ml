import tensorflow as tf
import tensorflow_datasets as tfds
import os
import official.nlp as nlp
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
from official.nlp import bert
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import json
import callback_utils

tensorboard_path = 'logs/bert_classifier_glue_mrpc'
model_path = 'models/text/bert_classifier_glue_mrpc'

# has checkpoint and vocab
# gs is not implemented on windows, download the files and save them locally
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
hub_model_name = "bert_en_uncased_L-12_H-768_A-12"
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


def bert_encode(glue_dict, tokenizer):
    def encode_sentence(s):
        tokens = list(tokenizer.tokenize(s.numpy()))
        tokens.append('[SEP]')
        return tokenizer.convert_tokens_to_ids(tokens)

    sentence1s = tf.ragged.constant([encode_sentence(s)
                                     for s in glue_dict['sentence1']])
    sentence2s = tf.ragged.constant([encode_sentence(s)
                                     for s in glue_dict['sentence2']])
    # print(sentence1s.shape) #(3668, None)
    # print(sentence2s.shape) #(3668, None)
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1s.shape[0]
    # (3668)
    # each input is [CLS] SENTENCE1 [SEP] SENTENCE2 [SEP] [PAD] ... [PAD]
    input_word_ids = tf.concat([cls, sentence1s, sentence2s], axis=-1)
    # print(input_word_ids.shape) #(3668, None)
    # plt.pcolormesh(input_word_ids.to_tensor())
    # plt.show()
    # t = input_word_ids.to_tensor()
    # for i in range (0, t.shape[0]):
    #     # print(t[i, :])
    #     print(tokenizer.convert_ids_to_tokens(t[i, :].numpy()))
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    # print(input_mask.shape) #(3668, 103)
    # plt.pcolormesh(input_mask)
    # plt.show()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1s)
    type_s2 = tf.ones_like(sentence2s)
    # since sentence1s and sentence2s are ruggedtensor, we'll need to call to_tensor() to convert it to a regular
    # tensor, until then they'll not have length of a sentence
    # after converted, the maxlength will be the output tensor length
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()
    # plt.pcolormesh(input_type_ids)
    # plt.show()

    inputs = {
        # [CLS] SENTENCE1 [SEP] SENTENCE2 [SEP] [PAD] ... [PAD]
        'input_word_ids': input_word_ids.to_tensor(),
        # 1     1 1 1 1   1     1 1 1 1   1     0 0 0...      0
        'input_mask': input_mask,
        # 0 0 0...        0     1 1 1 1   1     0 0 0...      0
        'input_type_ids': input_type_ids
    }
    return inputs


def get_tokenizer():
    return bert.tokenization.FullTokenizer(
        vocab_file=os.path.join("bert_vocab.txt"),
        do_lower_case=True)


def get_data():
    # General Language Understanding Evaluation benchmark
    # The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically
    # extracted from online news sources, with human annotations for whether the sentences in the pair are semantically
    # equivalent.
    glue, info = tfds.load('glue/mrpc', with_info=True, batch_size=-1)
    # print(ds.keys())
    # print(info)
    # no supervised keys
    # FeaturesDict({
    #     'idx': tf.int32,
    #     'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=2),
    #     'sentence1': Text(shape=(), dtype=tf.string),
    #     'sentence2': Text(shape=(), dtype=tf.string),
    # })

    # dataset has 4 keys, each mapping to a tensor of [3668] records
    # for key, value in glue_train.items():
    #     print(f"{key:9s}: {value[0].numpy()}")

    tokenizer = get_tokenizer()

    # print("vocab size: {}".format(len(tokenizer.vocab))) # 30522
    # tokens = tokenizer.tokenize("yo sup dude")
    # print(tokens)
    # ids = tokenizer.convert_tokens_to_ids(tokenizer)
    # print(ids)

    glue_train = bert_encode(glue['train'], tokenizer)
    glue_train_labels = glue['train']['label']
    # for key, value in glue_train.items():
    #     print(f'{key:15s} shape: {value.shape}')
    # print(f'glue_train_labels shape: {glue_train_labels.shape}')
    # input_word_ids  shape: (3668, 103)
    # input_mask      shape: (3668, 103)
    # input_type_ids  shape: (3668, 103)
    # glue_train_labels shape: (3668,)

    glue_validation = bert_encode(glue['validation'], tokenizer)
    glue_validation_labels = glue['validation']['label']

    glue_test = bert_encode(glue['test'], tokenizer)
    glue_test_labels = glue['test']['label']
    return glue_train, glue_train_labels, glue_validation, glue_validation_labels, glue_test, glue_test_labels


def get_model():
    # bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
    # config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
    config_dict = {'attention_probs_dropout_prob': 0.1,
                   'hidden_act': 'gelu',
                   'hidden_dropout_prob': 0.1,
                   'hidden_size': 768,
                   'initializer_range': 0.02,
                   'intermediate_size': 3072,
                   'max_position_embeddings': 512,
                   'num_attention_heads': 12,
                   'num_hidden_layers': 12,
                   'type_vocab_size': 2,
                   'vocab_size': 30522}

    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    ###### create model from gs_folder_bert
    # # num_labels=2 essentially creates a head
    # bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)
    # tf.keras.utils.plot_model(bert_classifier, to_file='bert_classifier.png', show_shapes=True, dpi=48)
    # tf.keras.utils.plot_model(bert_encoder, to_file='bert_encoder.png', show_shapes=True, dpi=48)
    # # print(len(bert_classifier.trainable_variables))
    # # print(len(bert_encoder.trainable_variables))
    # # bert_classifier is the entire model, bert_encoder is part of it
    # # it takes the 3 vectors as input and output 2-length logits on whether the two sentence has same semantic meaning
    #
    #
    # # got stuck restoring ckpt
    # checkpoint = tf.train.Checkpoint(model=bert_encoder)
    # checkpoint.restore(
    #     os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
    # return bert_classifier, bert_encoder
    ##### create model from tf hub

    # hub_encoder = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{hub_model_name}/2",
    #                              trainable=True)
    hub_classifier, hub_encoder = bert.bert_models.classifier_model(
        # Caution: Most of `bert_config` is ignored if you pass a hub url.
        bert_config=bert_config, hub_module_url=hub_url_bert, num_labels=2)
    return hub_classifier, hub_encoder


def train_model():
    glue_train, glue_train_labels, glue_validation, glue_validation_labels, glue_test, glue_test_labels = get_data()
    bert_classifier, bert_encoder = get_model()

    epochs = 3
    batch_size = 32
    eval_batch_size = 32

    train_data_size = len(glue_train_labels)
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(
        2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
    # official.nlp.optimization.AdamWeightDecay

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    bert_classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    bert_classifier.fit(
        glue_train, glue_train_labels,
        validation_data=(glue_validation, glue_validation_labels),
        batch_size=32,
        epochs=epochs,
        callbacks=[callback_utils.create_tensorboard_callback(tensorboard_path)]
    )
    tf.saved_model.save(bert_classifier, model_path)


def reload_model_and_test():
    reloaded = tf.saved_model.load(model_path)
    my_examples = bert_encode(
        glue_dict={
            'sentence1': [
                'The rain in Spain falls mainly on the plain.',
                'Look I fine tuned BERT.'],
            'sentence2': [
                'It mostly rains on the flat lands of Spain.',
                'Is it working? This does not match.']
        },
        tokenizer=get_tokenizer())

    reloaded_result = reloaded([my_examples['input_word_ids'],
                                my_examples['input_mask'],
                                my_examples['input_type_ids']], training=False)

    print(reloaded_result.numpy())


def main():
    train_model()


if __name__ == '__main__':
    main()
