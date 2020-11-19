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


# MARK: - Create from tensor
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
    print("Value type:", dataset.element_spec.value_type)    # <class 'tensorflow.python.framework.ops.Tensor'>


def test_nums_3():
    dataset = tf.data.Dataset.from_tensor_slices([tf.random.uniform([3, 3])] * 10)
    for tensor in dataset:    # 10 tensors of shape (3, 3)
        print(f"Shape: {tensor.shape}")

    print("Element spec:", dataset.element_spec)    # TensorSpec(shape=(3, 3), dtype=tf.float32, name=None)


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
    print("Value type 0:", zipped_dataset.element_spec[0].value_type)    # Tensor


# MARK: - Built in datasets
def test_mnist():
    """
    The MNIST dataset contains images of `int` values 0 ~ 255.
    """
    train, test = tf.keras.datasets.mnist.load_data()
    
    print("Train:", type(train), len(train))    # <class 'tuple'> 2
    train_images, train_labels = train
    print(f"Train images: type: {type(train_images)}, len: {len(train_images)}, max: {np.max(train_images)}, min: {np.min(train_images)}, dtype: {train_images.dtype}")    # type: <class 'numpy.ndarray'>, len: 60000, max: 255, min: 0, dtype: uint8
    print(f"Train labels: type: {type(train_labels)}, len: {len(train_labels)}, max: {np.max(train_labels)}, min: {np.min(train_labels)}, dtype: {train_labels.dtype}")    # type: <class 'numpy.ndarray'>, len: 60000, max: 9, min: 0, dtype: uint8

    print("Test:", type(test), len(test))    # <class 'tuple'> 2
    test_images, test_labels = test
    print("Test images:", type(test_images), len(test_images))
    print("Test labels:", type(test_labels), len(test_labels))

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    print("Element spec:", dataset.element_spec)    # (TensorSpec(shape=(28, 28), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.uint8, name=None))


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    np.set_printoptions(precision=4)

    # test_nums_1()
    # test_nums_2()
    # test_nums_3()

    # test_nums_tuple()

    # test_zip()

    test_mnist()
