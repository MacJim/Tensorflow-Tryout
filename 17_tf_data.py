"""
Source: https://www.tensorflow.org/guide/data
"""

import os
import datetime

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
import numpy as np


def test_nums_1():
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(6, dtype=tf.float32))
    print("Dataset:")    # 6 tensors from 0.0 to 5.0
    for num in dataset:
        print(num)

    sum = dataset.reduce(0.0, lambda state, value: state + value)    # Sum: 15.0; 0.0 here is the initial state
    print("Sum:", sum)


def test_nums_2():
    dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([6, 10]))
    print("Dataset:")    # 6 tensors of shape (10,)
    for tensor in dataset:
        print(tensor)

    print("Element spec:", dataset.element_spec)    # TensorSpec(shape=(10,), dtype=tf.float32, name=None)


def test_nums_tuple():
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.range(4),
        tf.range(20, 24),
        tf.random.uniform([4, 100], minval=4, maxval=100, dtype=tf.int32)
    ))
    for tensor_tuple in dataset:
        print(f"Tuple length: {len(tensor_tuple)}")    # 3
        print(tensor_tuple[0])    # Scalar
        print(tensor_tuple[1])    # Scalar
        print(tensor_tuple[2])    # Shape (100,)
        print()

    print("Element spec:", dataset.element_spec)


def test_zip():
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.range(6))
    dataset2 = tf.data.Dataset.from_tensor_slices(tf.range(20, 26))
    zipped_dataset = tf.data.Dataset.zip((dataset1, dataset2))    # (0, 20) to (5, 25)
    for tensor_tuple in zipped_dataset:
        print(f"Tuple length: {len(tensor_tuple)}")    # 2
        print(tensor_tuple[0])    # Scalar
        print(tensor_tuple[1])    # Scalar
        print()

    print("Element spec:", zipped_dataset.element_spec)


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    np.set_printoptions(precision=4)

    # test_nums_1()
    # test_nums_2()

    # test_nums_tuple()

    test_zip()
