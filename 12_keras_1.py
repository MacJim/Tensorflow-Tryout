"""
Source: https://www.tensorflow.org/guide/intro_to_modules#keras_models_and_layers

Note:
A raw `tf.Module` nested inside a Keras layer or model will not get its variables collected for training or saving.
Instead, nest Keras layers inside of Keras layers.
"""

import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
from tensorflow import keras


# MARK: - Module definitions
class LazyDenseReLU (keras.layers.Layer):
    def __init__(self, out_features: int):
        super().__init__()

        self.out_features = out_features

    def build(self, input_shape):
        """
        Create states (weights) of the layer in the `build` method.

        The model is "built" when an input's given.
        The model contains no variables if it has not been built.

        The `build` method is only called once.
        Thus, the input shapes are fixed on the first call.
        """
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.out_features]), name="w")
        self.b = tf.Variable(tf.zeros([self.out_features]), name="b")


    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        In a Keras layer, use `call` instead of `__call__`.
        """
        x = x @ self.w + self.b
        return tf.nn.relu(x)


class TestModule (keras.Model):
    def __init__(self):
        super().__init__()

        self.dense_1 = LazyDenseReLU(out_features=3)
        self.dense_2 = LazyDenseReLU(out_features=2)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Use `call` instead of `__call__` in Keras models as well.
        """
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    # MARK: Create model
    model = TestModule()
    print("Sub modules:", model.submodules)

    # MARK: Infer
    input = tf.random.normal((1, 3))
    output = model(input)
    print("Output:", output)
