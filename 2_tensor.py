import sys
import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
import numpy as np


# MARK: - Create tensor with scaler or list
def test_creation_1():
    """
    - Integers are parsed as `int32`
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


def test_creation_2():
    """
    - Floats are parsed as `float32`
    """
    a = tf.constant(2.16)    # a: 2.1600000858306885
    print(f"a: {a}")    # tf.Tensor(2.16, shape=(), dtype=float32)
    print(a)    # tf.Tensor([[1. 2.]], shape=(1, 2), dtype=float32)

    # Force `dtype`.
    b = tf.constant([[1, 2]], dtype=tf.float32)
    print(f"b: {b}")    # b: [[1. 2.]]
    print(b)    # tf.Tensor([[1. 2.]], shape=(1, 2), dtype=float32)


def test_creation_3():
    """
    Can use tuples instead of lists when creating a tensor.
    """
    a = tf.constant((
        (1, 2),
        (3, 4)
    ))
    print(f"a: {a}")

    b = tf.constant((
        (1, 2),
        (3, 4)
    ), dtype=tf.float32)
    print(f"b: {b}")

    c = tf.constant((
        (1.0, 2.0),
        (3.0, 4.0)
    ))
    print(f"c: {c}")


# MARK: - Special tensor creation functions
def test_range_creation():
    """
    Range.
    """
    a = tf.range(5)
    print("a:", a)    # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)

    b = tf.range(1, 7, dtype=tf.float32)
    print("b:", b)    # tf.Tensor([1. 2. 3. 4. 5. 6.], shape=(6,), dtype=float32)

    c = tf.range(1, 7, 2, dtype=tf.float32)
    print("c:", c)    # tf.Tensor([1. 3. 5.], shape=(3,), dtype=float32)

    d = tf.range(1, 8, 2, dtype=tf.float32)
    print("d:", d)    # tf.Tensor([1. 3. 5. 7.], shape=(4,), dtype=float32)

    e = tf.range(6, 0, -1)    # Reversed order
    print("e:", e)    # tf.Tensor([6 5 4 3 2 1], shape=(6,), dtype=int32)

    e_alt = tf.reverse(b, axis=tf.constant([-1]))    # Another way to flip the tensor (this function is called `flip` in torch)
    print("e_alt:", e_alt)    # tf.Tensor([6. 5. 4. 3. 2. 1.], shape=(6,), dtype=float32)


# TODO: Tensors are immutable
def test_mutable():
    pass


# tf.get_logger().setLevel("ERROR")    # Does not work.

# test_creation_1()
# test_creation_2()
# test_creation_3()

test_range_creation()
