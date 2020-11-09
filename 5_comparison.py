import os
import sys

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_element_wise_larger_smaller_1():
    """
    Element-wise comparisons.
    """
    a = tf.constant([
        [1, 2],
        [2, 3]
    ], dtype=tf.float32)
    b = tf.constant([
        [3, 2],
        [2, 1]
    ], dtype=tf.float32)

    result1 = a > b
    result2 = a >= b
    result3 = tf.math.maximum(a, b)

    print(f"a > b: {result1}")    # [[False False] [False  True]]
    print(f"a >= b: {result2}")    # [[False  True] [ True  True]]
    print(f"tf.math.maximum(a, b): {result3}")    # [[3. 2.] [2. 3.]]


def test_element_wise_larger_smaller_2():
    """
    Can compare scalar tensors.
    """
    a = tf.constant(5)
    b = tf.constant(6)

    result1 = a > b
    result2 = tf.math.maximum(a, b)

    print(f"a > b: {result1}, {type(result1)}, {result1.shape}")    # False, <class 'tensorflow.python.framework.ops.EagerTensor'>, ()
    print(f"tf.math.maximum(a, b): {result2}, {type(result2)}, {result2.shape}")    # 6, <class 'tensorflow.python.framework.ops.EagerTensor'>, ()


def test_element_wise_larger_smaller_3():
    """
    Cannot compare tensors of different `dtype`s.
    Does not work even between (`int32`, `int64`) or (`float32`, `float64`).
    """
    a = tf.constant([
        [1, 2],
        [2, 3]
    ], dtype=tf.float32)
    b = tf.constant([
        [3, 2],
        [2, 1]
    ], dtype=tf.int32)
    c = tf.cast(a, tf.int64)
    d = tf.cast(b, tf.float64)

    try:
        result_ab_1 = a > b
    except:
        print(f"Cannot compare a and b: exception: {sys.exc_info()}")    # (<class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>, InvalidArgumentError(), <traceback object at 0x7fd74ef59088>)

    try:
        result_ab = tf.math.maximum(a, b)
    except:
        print(f"Cannot get max of a and b: exception: {sys.exc_info()}")    # (<class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>, InvalidArgumentError(), <traceback object at 0x7fd74ef59048>)

    try:
        result_bc = b > c
    except:
        print(f"Cannot compare b and c: exception: {sys.exc_info()}")    # (<class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>, InvalidArgumentError(), <traceback object at 0x7fd74ef59088>)

    try:
        result_ad = a > d
    except:
        print(f"Cannot compare a and d: exception: {sys.exc_info()}")    # (<class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>, InvalidArgumentError(), <traceback object at 0x7fd74ef59048>)


def test_element_wise_larger_smaller_different_shapes():
    """
    Do element wise comparisons on tensors of different shapes.
    """
    a = tf.range(6)
    b = tf.reshape(a, (2, -1))

    print("a:", a)
    print("b:", b)

    try:
        result1 = a > b    # tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [6] vs. [2,3] [Op:Greater]
        print(result1)
    except Exception as e:
        print(sys.exc_info())
        print(e)


# test_element_wise_larger_smaller_1()
# test_element_wise_larger_smaller_2()
# test_element_wise_larger_smaller_3()

test_element_wise_larger_smaller_different_shapes()
