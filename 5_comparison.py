import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_element_wise_larger_smaller_1():
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
    pass


test_element_wise_larger_smaller_1()
