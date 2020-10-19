import os
import sys

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


def test_multiply_1():
    a = 6
    b = tf.constant(6)
    c = tf.constant([6, 6, 6])
    d = tf.constant([1, 2, 3])

    # Every operation below yields the same result: tf.Tensor([ 6 12 18], shape=(3,), dtype=int32)
    print(a * d)
    print(b * d)
    print(c * d)
    print(d * a)
    print(d * b)
    print(d * c)


def test_multiply_2():
    """
    Can use the `*` operator on a vertical vector and a horizontal vector.
    In this case, the `*` operator behaves as a `@` (matrix multiplication) operator.

    I think it's better to use `@` to prevent ambiguity.
    """
    a = tf.range(3)
    a = tf.reshape(a, (3, 1))
    b = tf.range(4)
    print(a * b)    # [[0 0 0 0] [0 1 2 3] [0 2 4 6]] Somehow this becomes matrix multiplication.
    # print(a @ b)    # InvalidArgumentError: b is not a matrix

    c = tf.reshape(b, (1, 4))
    print(a @ c)    # [[0 0 0 0] [0 1 2 3] [0 2 4 6]]

    d = tf.range(3)
    e = tf.range(4)
    e = tf.reshape(e, (4, 1))
    print(d * e)    # [[0 0 0] [0 1 2] [0 2 4] [0 3 6]] Somehow the `*` operator swapped `d` and `e` for matrix multiplication.
    
    f = tf.reshape(d, (1, 3))
    print(e @ f)    # [[0 0 0] [0 1 2] [0 2 4] [0 3 6]]


def test_multiply_3():
    """
    Cannot use the `*` operator on 2 horizontals vectors, or 2 vertical vectors.
    """
    a = tf.range(3)
    b = tf.range(4)
    # print(a * b)    # InvalidArgumentError: Incompatible shapes: [3] vs. [4] [Op:Mul]

    c = tf.reshape(a, (1, 3))
    d = tf.reshape(b, (1, 4))
    # print(c * d)    # InvalidArgumentError: Incompatible shapes: [1,3] vs. [1,4] [Op:Mul]


def test_multiply_4():
    """
    Cannot use the `*` operator on 2 matrices.
    """
    a = tf.range(12)
    b = tf.reshape(a, (4, 3))
    c = tf.reshape(a, (3, 4))
    # print(b * c)    # InvalidArgumentError: Incompatible shapes: [4,3] vs. [3,4] [Op:Mul]
    print(b @ c)    # [[ 20  23  26  29] [ 56  68  80  92] [ 92 113 134 155] [128 158 188 218]]


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

    Unless `keepdims` is `True`, the rank (dimension) of the tensor is reduced by 1 for each entry in axis.
    """
    a = tf.constant(6)
    print(f"a reduce_max default: {tf.reduce_max(a)}")    # 6
    print(f"a reduce_max keep dims: {tf.reduce_max(a, keepdims=True)}")    # 6

    b = tf.constant([3, 2, 1], dtype=tf.float32)
    print(f"b reduce_max default: {tf.reduce_max(b)}")    # 3.0  If `axis` is None (default), all dimensions are reduced, and a tensor with a single element is returned.
    print(f"b reduce_max -1: {tf.reduce_max(b, axis=-1)}")    # 3.0
    print(f"b reduce_max default keep dims: {tf.reduce_max(b, keepdims=True)}")    # [3.]
    print(f"b reduce_max -1 keep dims: {tf.reduce_max(b, axis=-1, keepdims=True)}")    # [3.]

    c = tf.constant([
        [7, 8, 9],
        [1, 2, 3],
        [4, 5, 6],
    ])
    print(f"c reduce_max default: {tf.reduce_max(c)}")    # 9
    print(f"c reduce_max 0: {tf.reduce_max(c, axis=0)}")    # [7 8 9]
    print(f"c reduce_max 1: {tf.reduce_max(c, axis=1)}")    # [9 3 6]
    print(f"c reduce_max -1: {tf.reduce_max(c, axis=-1)}")    # [9 3 6]
    print(f"c reduce_max default keep dims: {tf.reduce_max(c, keepdims=True)}")    # [[9]]
    print(f"c reduce_max -1 keep dims: {tf.reduce_max(c, axis=-1, keepdims=True)}")    # [[9] [3] [6]]
    print(f"c reduce_max 0 keep dims: {tf.reduce_max(c, axis=0, keepdims=True)}")    # [[7 8 9]]
    print(f"c reduce_max 1 keep dims: {tf.reduce_max(c, axis=1, keepdims=True)}")    # [[9] [3] [6]]

    d = tf.range(8)
    d = tf.reshape(d, (2, 2, 2))
    print("d:", d)    # [[[0 1] [2 3]]  [[4 5] [6 7]]]
    print(f"d reduce_max 0: {tf.reduce_max(d, axis=0)}")    # [[4 5] [6 7]]
    print(f"d reduce_max (0, 1): {tf.reduce_max(d, axis=(0, 1))}")    # [6 7]


def test_argmax():
    """
    Returns the index with the largest value across axes of a tensor.

    - May only specify a single axis (differnet from `reduce_max`)
    """
    # Cannot calculate on tensors without axes.
    a = tf.constant(6)
    try:
        print(f"a argmax default: {tf.argmax(a)}")
    except:
        print(f"Tensor without axes: {sys.exc_info()}")

    b = tf.range(6)
    b = tf.reshape(b, (2, 3))    # [[0 1 2] [3 4 5]]
    print(f"b argmax default: {tf.argmax(b)}")    # [1 1 1] Default axis is 0 (this is different from `reduce_max`)
    print(f"b argmax 0: {tf.argmax(b, axis=0)}")    # [1 1 1]
    print(f"b argmax 1: {tf.argmax(b, axis=1)}")    # [2 2]
    print(f"b argmax -1: {tf.argmax(b, axis=-1)}")    # [2 2]

    print(f"b argmax 0 int32: {tf.argmax(b, axis=0, output_type=tf.int32)}")    # Default output tensor dtype is `int64`

    # Returns the **smallest/first** index of the smallest value in case of ties.
    c = tf.ones((2, 3))
    c *= 6
    print(f"c argmax default: {tf.argmax(c)}")     # Always [0 0 0]
    print(f"c argmax -1: {tf.argmax(c, axis=-1)}")    # Always [0 0]

    
# MARK: - Main
# test_basics()

# test_multiply_1()
test_multiply_2()
# test_multiply_3()
# test_multiply_4()

# test_softmax()

# test_reduce_max()

# test_argmax()
