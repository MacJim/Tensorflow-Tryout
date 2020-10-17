import os
import sys

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_single_axis_scalar():
    a = tf.range(10)
    print(a[6])    # tf.Tensor(6, shape=(), dtype=int32) The result is a scalar.


def test_single_axis_sub_range():
    a = tf.range(10)
    print(a[:6])    # tf.Tensor([0 1 2 3 4 5], shape=(6,), dtype=int32)
    print(a[::2])    # tf.Tensor([0 2 4 6 8], shape=(5,), dtype=int32) Every other item.
    print(a[::-1])    # tf.Tensor([9 8 7 6 5 4 3 2 1 0], shape=(10,), dtype=int32) Reversed (similar to `tf.reverse`).


# MARK: Main
test_single_axis_scalar()

test_single_axis_sub_range()
