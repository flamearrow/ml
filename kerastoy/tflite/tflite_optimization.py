import tensorflow as tf
import text.tc_rnn as tc_rnn

lstm_model_lite_path = "lstm_model.tflite"


# convert the model from tc_rnn.py

def convert_lstm_model_to_lite(with_quantization=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(tc_rnn.model_path)
    if with_quantization:
        # default(wihtout any target_spec) is dynamic range quantization
        # at inference, the weights are converted back from 8-bit integer to floating poitn
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # add this line for float16 - optimized for for gpus
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    # model has float32[1, 1] as input and float32[1, 1] as output
    # with dynamic range 8-bit quantization, size is 677kb
    # with float16 quantization, size is 1.3mb
    # without quantization, size is 2.6mb
    with open(lstm_model_lite_path, "wb") as f:
        f.write(tflite_model)


def main():
    convert_lstm_model_to_lite(with_quantization=True)


if __name__ == "__main__":
    main()
