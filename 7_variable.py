import os
import sys

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_basics():
    x = tf.range(6)
    x = tf.reshape(x, (2, 3))

    y = tf.Variable(x)
    print(y)
    print(f"Shape: {y.shape}")    # (2, 3)
    print(f"dtype: {y.dtype}")    # <dtype: 'int32'>
    print(f"To numpy: {y.numpy()}")    # [[0 1 2] [3 4 5]]

    z = tf.convert_to_tensor(y)
    print(z)


def test_operations_1():
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


def test_operations_2():
    """
    Use `assign`, `assign_add`, and `assign_sub` to modify a `Variable`.
    """
    x = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
    print(x)    # Variable [1., 2., 3., 4.]
    x.assign(tf.range(6, 10, dtype=tf.float32))
    print(x)    # Variable [6., 7., 8., 9.]
    x.assign_add(tf.ones(x.shape, dtype=x.dtype))
    print(x)    # Variable [ 7.,  8.,  9., 10.]
    x.assign_sub(tf.ones(x.shape, dtype=x.dtype))
    print(x)    # Variable [6., 7., 8., 9.]


def test_memory():
    """
    Variables copied from tensors/variables don't share memory.
    """
    x = tf.range(6)

    y1 = tf.Variable(x)
    y2 = tf.Variable(y1)
    y3 = tf.Variable(y1)

    y1.assign(tf.ones(y1.shape, dtype=tf.int32))
    y2.assign(tf.range(1, 7))
    y3.assign_sub(tf.ones(y3.shape, dtype=y3.dtype))    # Note that the default dtype for `ones` is `tf.dtypes.float32`.

    print(f"x: {x}")    # [0 1 2 3 4 5]
    print(f"y1: {y1}")    # Variable [1, 1, 1, 1, 1, 1]
    print(f"y2: {y2}")    # Variable [1, 2, 3, 4, 5, 6]
    print(f"y3: {y3}")    # Variable [-1,  0,  1,  2,  3,  4]


if (__name__ == "__main__"):
    # test_basics()
    # test_operations_1()
    # test_operations_2()
    test_memory()
