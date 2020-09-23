import sys
import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
import numpy as np


def test_creation_0():
    """
    - Integers are treated as `int32`
    - Somehow f-strings do not print additional tensor info
    - TODO: What are tensors without axes used for?
    """
    # A tensor with no axes.
    a = tf.constant(4)
    print(f"a: {a}")    # a: 4
    print(a)    # tf.Tensor(4, shape=(), dtype=int32)
    # Shape objects are not tuples.
    print(f"a.shape: {a.shape}, {type(a.shape)}")    # a.shape: (), <class 'tensorflow.python.framework.tensor_shape.TensorShape'>
    try:
        print(f"a.shape[0]: {a.shape[0]}")
    except:
        print(f"a.shape[0] exception: {sys.exc_info()}")    # a.shape[0] exception: (<class 'IndexError'>, IndexError('list index out of range',), <traceback object at 0x7effa5dcbdc8>)

    b = tf.constant([4])
    print(f"b: {b}")    # b: [4]
    print(b)    # tf.Tensor([4], shape=(1,), dtype=int32)
    print(f"b.shape: {b.shape}, {type(b.shape)}")    # b.shape: (1,), <class 'tensorflow.python.framework.tensor_shape.TensorShape'>
    print(f"b.shape[0]: {b.shape[0]}")    # b.shape[0]: 1

    c = tf.constant([1, 2, 3])
    print(f"c: {c}")    # c: [1 2 3]
    print(c)    # tf.Tensor([1 2 3], shape=(3,), dtype=int32)


def test_mutable():
    # TODO: Tensors are immutable.
    pass


# tf.get_logger().setLevel("ERROR")    # Does not work.

test_creation_0()
