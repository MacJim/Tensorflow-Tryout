"""
https://www.tensorflow.org/guide/keras/sequential_model

A Sequential model is NOT APPROPRIATE when:

- Your model has multiple inputs or multiple outputs
- Any of your layers has multiple inputs or multiple outputs
- You need to do layer sharing
- You want non-linear topology (e.g. a residual connection, a multi-branch model)

"""

import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def test_creation():
    model = keras.Sequential(
        [
            keras.layers.Dense(2, activation="relu", name="layer1"),
            keras.layers.Dense(3, activation="relu", name="layer2"),
            keras.layers.Dense(4, name="layer3"),
        ]
    )
    model.pop()
    model.add(keras.layers.Dense(4, name="layer3-2"))
    print("Layers:")
    print(model.layers)
    
    x = tf.ones((3, 3))
    y = model(x)
    print(y)


def test_build_1():
    model = keras.Sequential(
        [
            keras.layers.Dense(2, activation="relu", name="layer1"),
            keras.layers.Dense(3, activation="relu", name="layer2"),
            keras.layers.Dense(4, name="layer3"),
        ]
    )

    print("Layer 0 weights before input:", model.layers[0].weights)    # [] Weights are created the first time the model is called on an input.
    # print(model.weights)    # ValueError: Weights for model sequential have not yet been created. Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.
    # model.summary()    # ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.

    x = tf.ones((1, 4))    # Build the model by supplying an input.
    y = model(x)
    print("Layer 0 weights after input:", model.layers[0].weights)    # shape=(2,)
    model.summary()


def test_build_2():
    model = keras.Sequential(
        [
            # Let the model know the input shape from the start.
            # This is NOT a layer.
            # We're also not specifying the 0th dimension (should be the batch size, right).
            keras.Input((4,)),
            keras.layers.Dense(2, activation="relu", name="layer1"),
            keras.layers.Dense(3, activation="relu", name="layer2"),
            keras.layers.Dense(4, name="layer3"),
        ]
    )
    print("Layers:", model.layers)    # Does not contain `Input`.
    model.summary()


def test_build_3():
    model = keras.Sequential(
        [
            keras.layers.Dense(2, activation="relu", name="layer1", input_shape=(4,)),    # Specify input shape as an argument.
            keras.layers.Dense(3, activation="relu", name="layer2"),
            keras.layers.Dense(4, name="layer3"),
        ]
    )
    print("Layers:", model.layers)    # Does not contain `Input`.
    model.summary()


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    # test_creation()

    # test_build_1()
    # test_build_2()
    # test_build_3()
