import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_basics():
    i = tf.constant([
        [1, 0],
        [0, 1]
    ], dtype=tf.float32)

    a = tf.constant([
        [1, 2],
        [3, 4]
    ], dtype=tf.float32)

    b = tf.constant([
        [1, 1],
        [1, 1]
    ], dtype=tf.float32)

    # Add.
    add_a_b_1 = a + b
    add_a_b_2 = tf.add(a, b)

    print("Add:")
    print("a + b:", add_a_b_1)    # [[2. 3.] [4. 5.]]
    print("tf.add(a, b):", add_a_b_2)    # [[2. 3.] [4. 5.]]

    # Element wise multiplication.
    mul_a_b_1 = a * b
    mul_a_b_2 = tf.multiply(a, b)

    print("Element wise mul:")
    print("a * b:", mul_a_b_1)    # [[1. 2.] [3. 4.]]
    print("tf.multiply(a, b):", mul_a_b_2)    # [[1. 2.] [3. 4.]]

    # Matrix mul.
    mat_mul_a_b_1 = a @ b
    mat_mul_a_b_2 = tf.matmul(a, b)
    mat_mul_a_i_1 = a @ i
    mat_mul_a_i_2 = tf.matmul(a, i)

    print("Matrix mul:")
    print("a @ b:", mat_mul_a_b_1)    # [[3. 3.] [7. 7.]]
    print("tf.matmul(a, b):", mat_mul_a_b_2)    # [[3. 3.] [7. 7.]]
    print("a @ i:", mat_mul_a_i_1)    # [[1. 2.] [3. 4.]]
    print("tf.matmul(a, i):", mat_mul_a_i_2)    # [[1. 2.] [3. 4.]]


def test_softmax():
    a = tf.constant([
        [1, 1],
        [2, 2]
    ], dtype=tf.float32)
    
    print(f"a softmax default: {tf.nn.softmax(a)}")    # [[0.5 0.5] [0.5 0.5]] Default axis is -1.
    print(f"a softmax 0: {tf.nn.softmax(a, axis=0)}")    # [[0.26894143 0.26894143] [0.7310586  0.7310586 ]]
    print(f"a softmax 1: {tf.nn.softmax(a, axis=1)}")    # [[0.5 0.5] [0.5 0.5]]
    print(f"a softmax -1: {tf.nn.softmax(a, axis=-1)}")    # [[0.5 0.5] [0.5 0.5]]

    
# test_basics()
test_softmax()
