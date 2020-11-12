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


# MARK: - Constants
SAVED_MODEL_DIR_NAME = "saved_models/12"
SAVED_MODEL_WEIGHTS_FILENAME = "checkpoints/12"


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


# MARK: - Functional API
def get_functional_model():
    inputs = keras.Input(shape=(3,))    # We're not specifying all dimensions here.
    x = LazyDenseReLU(out_features=3)(inputs)
    x = LazyDenseReLU(out_features=2)(x)

    functional_model: keras.Model = keras.Model(inputs=inputs, outputs=x, name="mj_functional_model")
    # print("Model summary:")
    print(functional_model.summary(line_length=100))    # The default line length clips some of the description.

    return functional_model


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    # MARK: Create/restore model
    model = TestModule()
    # model = get_functional_model()

    model.load_weights(SAVED_MODEL_WEIGHTS_FILENAME)
    # model = keras.models.load_model(SAVED_MODEL_DIR_NAME)

    print("Sub modules:", model.submodules)

    # MARK: Infer
    input = tf.ones((1, 3))
    output = model(input)
    print("Output:", output)

    # MARK: Save model
    # model.save(SAVED_MODEL_DIR_NAME)    # This saves not only the model, but other states (e.g. optimizer state) as well.
    # model.save_weights(SAVED_MODEL_WEIGHTS_FILENAME)    # This only saves the weights.
