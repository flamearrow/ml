import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import toys.custom_model as cm

lstm_model_path = "lstm_model"
style_transoform_model_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# manually downloaded from style_transoform_model_dir
style_transform_model_path = "style_transform"


# use get_config to return the layer/model config, used to recreate the layer/model without weights
def keras_layer_toy():
    layer = tf.keras.layers.Dense(5, activation='relu')
    layer_config = layer.get_config()
    for index, item in enumerate(layer_config):
        print("{} : {}".format(item, layer_config[item]))
    # name: dense
    # trainable: True
    # dtype: float32
    # units: 5
    # activation: relu
    # use_bias: True
    # kernel_initializer: {'class_name': 'GlorotUniform', 'config': {'seed': None}}
    # bias_initializer: {'class_name': 'Zeros', 'config': {}}
    # kernel_regularizer: None
    # bias_regularizer: None
    # activity_regularizer: None
    # kernel_constraint: None
    # bias_constraint: None


# save and recreate a sequential/functional/exteded model using get_config() and from_config()
def keras_model_toy():
    model = cm.create_sequential_toy_model()
    # model.get_config():
    # {"layers": [list of layer config]}
    model_config = model.get_config()
    model2 = keras.Sequential.from_config(model_config)

    # model = cm.create_functional_model()
    # model.get_config():
    # {"layers": [list of layer config]",
    #  "input_layers":[list of input layer config],
    #  "output_layers":[list of output layer config]}

    # model = cm.MyCNNModel(num_classes=23)
    # model_config = model.get_config()
    # for index, item in enumerate(model_config):
    #     print("{} : {}".format(item, model_config[item]))
    #
    # model2 = cm.MyCNNModel.from_config(model_config)


# returns a saved model format
def get_hub_model():
    return hub.load(style_transoform_model_handle)


def try_saved_model_from_hub():
    model = get_hub_model()
    # print(list(model.signatures.keys()))
    # infer is a ConcreteFunction, an eagerly-executing wrapper around a tf.Graph.
    infer = model.signatures['serving_default']
    print(infer.structured_outputs)
    # {'output_0': TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name='output_0')}


def try_saved_model_from_hub():
    keras_layer = hub.KerasLayer(style_transoform_model_handle)
    # can use this to build other keras models


def convert_style_transform_to_lite():
    # hub_saved_model = get_hub_model()
    # tf.lite.TFLiteConverter.from_saved_model()
    # from concrete functions
    # saved_model = tf.saved_model.load(style_transform_model_path)
    # this doesn't work
    # converter = tf.lite.TFLiteConverter.from_concrete_functions([hub_saved_model.signatures['serving_default']])

    converter = tf.lite.TFLiteConverter.from_saved_model(style_transform_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("style_transfer.tflite", "wb") as f:
        f.write(tflite_model)


def load_from_local_file():
    tf.saved_model.load(lstm_model_path)


def main():
    convert_style_transform_to_lite()
    # load_from_local_file()
    # keras_layer_toy()
    # keras_model_toy()


if __name__ == "__main__":
    main()
