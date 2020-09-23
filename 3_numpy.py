import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
import numpy as np


def test_numpy_1():
    # 2 ways to convert from tensor to array.
    tensor1 = tf.constant([[1, 2]])
    print(f"tensor 1: {tensor1}")
    array11 = tensor1.numpy()
    array12 = np.array(tensor1)
    # The 2 arrays have different IDs.
    print(f"array 1-1: {array11}, {id(array11)}")    # [[1 2]], 140174015809856
    print(f"array 1-2: {array12}, {id(array12)}")    # [[1 2]], 140172777972000
    print(array11.dtype, array12.dtype)    # int32 int32

    tensor2 = tf.constant([[1, 2]])
    print(f"tensor 2: {tensor2}")
    array21 = tensor2.numpy()
    array22 = tensor2.numpy()
    # The 2 arrays have different IDs.
    print(f"array 2-1: {array21}, {id(array21)}")   # [[1 2]], 140345640431824
    print(f"array 2-2: {array22}, {id(array22)}")    # [[1 2]], 140345640431824

    tensor3 = tf.constant([[1, 2]])
    print(f"tensor 3: {tensor3}")
    array31 = np.array(tensor3)
    array32 = np.array(tensor3)
    # The 2 arrays have different IDs.
    print(f"array 3-1: {array31}, {id(array31)}")    # [[1 2]], 140134118117904
    print(f"array 3-2: {array32}, {id(array32)}")    # [[1 2]], 140134118117984

def test_numpy_2():
    # TODO: tensor -> array -> change array value -> check tensor value
    pass


test_numpy_1()