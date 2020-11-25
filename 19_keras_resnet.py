"""
Source: https://keras.io/api/applications/resnet/
"""
import os
import multiprocessing
import typing

import tensorflow as tf
from tensorflow import keras


# MARK: - Constants
SUMMARY_LINE_LEN: typing.Final = 132
RESNET_DEFAULT_INPUT_SHAPE: typing.Final = (224, 224, 3)
ALTERNATIVE_INPUT_SHAPE: typing.Final = (256, 256, 3)
N_CLASSES: typing.Final = 128


# MARK: - Print summary
def test_resnet50_layers():
    model = keras.applications.ResNet50()
    print(f"Layers: count: {len(model.layers)}, type: {type(model.layers)}")    # Layers: count: 177, type: <class 'list'>
    for layer in model.layers:
        print(layer)


def test_last_layer_softmax():
    """
    It seems that softmax is applied to the last (top) dense layer (by default).

    Yet I cannot find the softmax layer in the `.summary()`.
    """
    batch_size = 6

    # ResNet
    input1 = tf.random.uniform((batch_size,) + RESNET_DEFAULT_INPUT_SHAPE, minval=0.0, maxval=1.0)
    print(f"Input 1: min: {tf.reduce_min(input1).numpy()}, max: {tf.reduce_max(input1).numpy()}, mean: {tf.reduce_mean(input1).numpy()}")

    model11 = keras.applications.ResNet50()
    output11 = model11(input1)
    print(f"Output 11: min: {tf.reduce_min(output11).numpy()}, max: {tf.reduce_max(output11).numpy()}, mean: {tf.reduce_mean(output11).numpy()}, sum: {tf.reduce_sum(output11).numpy()}")    # Sum is always float(batch_size)

    model12 = keras.applications.ResNet50(weights=None, input_shape=RESNET_DEFAULT_INPUT_SHAPE, classes=N_CLASSES)
    output12 = model12(input1)
    print(f"Output 12: min: {tf.reduce_min(output12).numpy()}, max: {tf.reduce_max(output12).numpy()}, mean: {tf.reduce_mean(output12).numpy()}, sum: {tf.reduce_sum(output12).numpy()}")    # Sum is always float(batch_size)

    # ResNet v2
    input2 = tf.random.uniform((batch_size,) + RESNET_DEFAULT_INPUT_SHAPE, minval=-1.0, maxval=1.0)
    print(f"Input 2: min: {tf.reduce_min(input2).numpy()}, max: {tf.reduce_max(input2).numpy()}, mean: {tf.reduce_mean(input2).numpy()}")

    model21 = keras.applications.ResNet50V2()
    output21 = model21(input2)
    print(f"Output 21: min: {tf.reduce_min(output21).numpy()}, max: {tf.reduce_max(output21).numpy()}, mean: {tf.reduce_mean(output21).numpy()}, sum: {tf.reduce_sum(output21).numpy()}")    # Sum is always float(batch_size)

    model22 = keras.applications.ResNet50V2(weights=None, input_shape=RESNET_DEFAULT_INPUT_SHAPE, classes=N_CLASSES)
    output22 = model22(input2)
    print(f"Output 22: min: {tf.reduce_min(output22).numpy()}, max: {tf.reduce_max(output22).numpy()}, mean: {tf.reduce_mean(output22).numpy()}, sum: {tf.reduce_sum(output22).numpy()}")    # Sum is always float(batch_size)


def test_disable_last_layer_softmax():
    """
    Disable the last (top) dense layer's softmax by specifing `classifier_activation=None` or by copying the weights to a custom network.
    """
    batch_size = 6

    input1 = tf.random.uniform((batch_size,) + RESNET_DEFAULT_INPUT_SHAPE, minval=0.0, maxval=1.0)

    model11 = keras.applications.ResNet50(classifier_activation=None)    # Somehow the `classifier_activation` function is undocumented.
    output11 = model11(input1)
    print(f"Output 11: min: {tf.reduce_min(output11).numpy()}, max: {tf.reduce_max(output11).numpy()}, mean: {tf.reduce_mean(output11).numpy()}, sum: {tf.reduce_sum(output11).numpy()}")    # Sum is no longer `batch_size`.

    model12 = keras.Sequential([
        keras.applications.ResNet50(include_top=False),
        keras.layers.GlobalAveragePooling2D(name="avg_pool"),
        keras.layers.Dense(1000, name="predictions"),
    ])
    model12.set_weights(model11.get_weights())
    output12 = model12(input1)
    print(f"Output 12: min: {tf.reduce_min(output12).numpy()}, max: {tf.reduce_max(output12).numpy()}, mean: {tf.reduce_mean(output12).numpy()}, sum: {tf.reduce_sum(output12).numpy()}")

    input2 = tf.random.uniform((batch_size,) + RESNET_DEFAULT_INPUT_SHAPE, minval=-1.0, maxval=1.0)
    model21 = keras.applications.ResNet50V2(classifier_activation=None)
    output21 = model21(input2)
    print(f"Output 21: min: {tf.reduce_min(output21).numpy()}, max: {tf.reduce_max(output21).numpy()}, mean: {tf.reduce_mean(output21).numpy()}, sum: {tf.reduce_sum(output21).numpy()}")    # Sum is no longer `batch_size`.


# MARK: - Write summary to file
# MARK: ResNet 50
def test_resnet50_default():
    model = keras.applications.ResNet50()
    with open("19/resnet50_default.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=SUMMARY_LINE_LEN)


def test_resnet50_custom_input():
    """
    - When `include_top` is `True` (default) and `weights` is `"imagenet"` (default), input must has shape (224, 224, 3)
    """
    model = keras.applications.ResNet50(weights=None, input_shape=ALTERNATIVE_INPUT_SHAPE, classes=N_CLASSES)
    with open("19/resnet50_custom_input.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=SUMMARY_LINE_LEN)


def test_resnet50_include_top_false():
    """
    `include_top` is `False`.

    - No `Dense` at the end of the network
    - By default, no `GlobalAveragePooling2D` at the end of the network
      - This can be controlled using the `pooling` parameter
    - Can specify input shape (h, w, c)
      - Common sense: do not specify batch size
    """
    model = keras.applications.ResNet50(include_top=False, input_shape=ALTERNATIVE_INPUT_SHAPE)
    with open("19/resnet50_include_top_false.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=SUMMARY_LINE_LEN)


# MARK: ResNet 50 v2
def test_resnet50v2_default():
    model = keras.applications.ResNet50V2()
    with open("19/resnet50v2_default.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=SUMMARY_LINE_LEN)


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    # MARK: Print summary
    # test_resnet50_layers()

    # test_last_layer_softmax()
    test_disable_last_layer_softmax()

    # exit(0)

    # MARK: Run some of the test functions concurrently
    # test_functions = [
    #     test_resnet50_default, 
    #     test_resnet50_custom_input, 
    #     test_resnet50_include_top_false,
    #     test_resnet50v2_default,
    # ]
    # with multiprocessing.Pool(2) as pool:
    #     for f in test_functions:
    #         pool.apply_async(f)
