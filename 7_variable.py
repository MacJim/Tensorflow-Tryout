import os
import sys

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test1():
    x = tf.range(6)
    x = tf.reshape(x, (2, 3))

    y = tf.Variable(x)
    print(y)
    print(f"Shape: {y.shape}")    # (2, 3)
    print(f"dtype: {y.dtype}")    # <dtype: 'int32'>
    print(f"To numpy: {y.numpy()}")    # [[0 1 2] [3 4 5]]

    z = tf.convert_to_tensor(y)
    print(z)


def test2():
    """
    Using the following operations on a variable returns a tensor instead:

    - `tf.reshape`
    - `+`, `-`, `*`, `/`
    """
    x1 = tf.range(4)
    x1 = tf.reshape(x1, (2, 2))

    y = tf.Variable(x1)
    z = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
    z_reshape = tf.reshape(z, (2, 2))
    z_add = z + 1
    z_sub = z - 1
    z_mul = z * 2
    z_div = z / 2

    print(x1)    # Tensor [[0 1] [2 3]]
    print(y)    # Variable [[0, 1], [2, 3]]
    print(z)    # Variable [1., 2., 3., 4.]
    print(z_reshape)    # Tensor [[1. 2.] [3. 4.]]
    print(z_add)    # Tensor [2. 3. 4. 5.]
    print(z_sub)    # Tensor [0. 1. 2. 3.]
    print(z_mul)    # Tensor [2. 4. 6. 8.]
    print(z_div)    # Tensor [0.5 1.  1.5 2. ]


def test3():
    x = tf.range(6)

    # They don't share their memory.
    y1 = tf.Variable(x)
    y2 = tf.Variable(y1)
    y3 = tf.Variable(y1)

    y2.assign(tf.range(1, 7))
    y3.assign_add(tf.ones(y3.shape, dtype=y3.dtype))    # Note that the default dtype for `ones` is `tf.dtypes.float32`.

    print(f"y1: {y1}")    # Variable [0, 1, 2, 3, 4, 5]
    print(f"y2: {y2}")    # Variable [1, 2, 3, 4, 5, 6]
    print(f"y3: {y3}")    # Variable [1, 2, 3, 4, 5, 6]


if (__name__ == "__main__"):
    # test1()
    # test2()
    test3()
