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
    add_a_b_2 = tf.add(c, b)

    print("Add:")
    print("a + b:", add_a_b_1)    # [[2. 3.] [4. 5.]]
    print("tf.add(a, b):", add_a_b_2)    # [[2. 3.] [4. 5.]]

    # Element wise multiplication.
    mul_a_b_1 = c * b
    mul_a_b_2 = tf.multiply(c, b)

    print("Element wise mul:")
    print("a * b:", mul_a_b_1)    # [[1. 2.] [3. 4.]]
    print("tf.multiply(a, b):", mul_a_b_2)    # [[1. 2.] [3. 4.]]

    # Matrix mul.
    mat_mul_a_b_1 = c @ b
    mat_mul_a_b_2 = tf.matmul(c, b)
    mat_mul_a_i_1 = c @ i
    mat_mul_a_i_2 = tf.matmul(c, i)

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


def test_reduce_max():
    """
    Calculate the max values across a dimension.

    Unless `keepdims` is true, the rank (dimension) of the tensor is reduced by 1 for each entry in axis.
    """
    a = tf.constant(6)
    print(f"a reduce_max default: {tf.reduce_max(a)}")    # 6
    print(f"a reduce_max keep dims: {tf.reduce_max(a, keepdims=True)}")    # 6

    b = tf.constant([3, 2, 1], dtype=tf.float32)
    print(f"b reduce_max default: {tf.reduce_max(b)}")    # 3.0  If `axis` is None, all dimensions are reduced, and a tensor with a single element is returned.
    print(f"b reduce_max -1: {tf.reduce_max(b, axis=-1)}")    # 3.0
    print(f"b reduce_max default keep dims: {tf.reduce_max(b, keepdims=True)}")    # [3.]
    print(f"b reduce_max -1 keep dims: {tf.reduce_max(b, axis=-1, keepdims=True)}")    # [3.]

    c = tf.constant([
        [7, 8, 9],
        [1, 2, 3],
        [4, 5, 6],
    ])
    print(f"c reduce_max default: {tf.reduce_max(c)}")    # 9
    print(f"c reduce_max -1: {tf.reduce_max(c, axis=-1)}")    # [9 3 6]
    print(f"c reduce_max default keep dims: {tf.reduce_max(c, keepdims=True)}")    # [[9]]
    print(f"c reduce_max -1 keep dims: {tf.reduce_max(c, axis=-1, keepdims=True)}")    # [[9] [3] [6]]
    print(f"c reduce_max 0 keep dims: {tf.reduce_max(c, axis=0, keepdims=True)}")    # [[7 8 9]]
    print(f"c reduce_max 1 keep dims: {tf.reduce_max(c, axis=1, keepdims=True)}")    # [[9] [3] [6]]

    
# test_basics()
# test_softmax()
test_reduce_max()
