"""
Source: https://www.tensorflow.org/guide/intro_to_modules
"""

import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


class DenseReLU (tf.Module):
    """
    We ask for both `in_features` and `out_features` so that we can obtain the shape of `self.w` right away.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name="w")    # `normal` here means a normal distribution (mean 0.0, standard deviation 1.0).
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = x @ self.w + self.b
        return tf.nn.relu(x)


class LazyDenseReLU (tf.Module):
    """
    We don't ask for `in_features` and infer it when this module is first called.
    """
    def __init__(self, out_features: int):
        super().__init__()

        self.is_built = False
        self.out_features = out_features
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([x.shape[-1], self.out_features]), name="w")
            self.is_built = True

        x = x @ self.w + self.b
        return tf.nn.relu(x)


class TestModule (tf.Module):
    def __init__(self):
        super().__init__()

        # self.dense_1 = DenseReLU(in_features=3, out_features=3)
        # self.dense_2 = DenseReLU(in_features=3, out_features=2)
        self.dense_1 = LazyDenseReLU(out_features=3)
        self.dense_2 = LazyDenseReLU(out_features=2)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


model = TestModule()
print(f"Trainable variables: {model.trainable_variables}")
print(f"All variables: {model.variables}")
print(f"Submodules: {model.submodules}")    # 2 `__main__.DenseReLU object`s

input = tf.ones((1, 3))
input *= 3    # [[3. 3. 3.]]
# print(input)

output = model(input)
print(output)    # shape=(1, 2)
