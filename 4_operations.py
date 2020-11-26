import os
import sys
import math

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


# MARK: +-*/
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


def test_add_sub():
    a = tf.range(4)
    a = tf.reshape(a, (2, 2))

    b = a + 1
    c = a - 4

    print(a)    # [[0 1] [2 3]]
    print(b)    # [[1 2] [3 4]]
    print(c)    # [[-4 -3] [-2 -1]]


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


def test_power():
    a = tf.reshape(tf.range(4, dtype=tf.float32), (2, 2))
    b = tf.reshape(tf.range(3, -1, -1, dtype=tf.float32), (2, 2))
    c = a - b
    print(c)    # [[-3. -1.] [ 1.  3.]]

    # They are the same: [[9. 1.] [1. 9.]]
    print(tf.square(c))
    print(c ** 2)

    print(tf.norm(c))    # L2 norm: sqrt(20)


# MARK: - Softmax
def test_softmax():
    a = tf.constant([
        [1, 1],
        [2, 2]
    ], dtype=tf.float32)
    
    print(f"a softmax default: {tf.nn.softmax(a)}")    # [[0.5 0.5] [0.5 0.5]] Default axis is -1.
    print(f"a softmax 0: {tf.nn.softmax(a, axis=0)}")    # [[0.26894143 0.26894143] [0.7310586  0.7310586 ]]
    print(f"a softmax 1: {tf.nn.softmax(a, axis=1)}")    # [[0.5 0.5] [0.5 0.5]]
    print(f"a softmax -1: {tf.nn.softmax(a, axis=-1)}")    # [[0.5 0.5] [0.5 0.5]]


# MARK: - Max and argmax
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


# MARK: - Norm
def test_norm_1():
    """
    Tests `tf.norm`.
    """
    a = tf.constant([1, 1, 1, 1], dtype=tf.float32)

    b = tf.reshape(a, (2, 2))
    b_norm = tf.norm(b)
    print(b_norm)    # 2.0 = 4.0 / sqrt(2.0)

    c = tf.reshape(a, (2, 2, 1, 1))
    c_norm_0 = tf.norm(c, axis=0)
    print(c_norm_0)    # [[[1.4142135]] [[1.4142135]]] shape=(2, 1, 1)
    # c_norm_1 = tf.norm(c, axis=(1, 2, 3))
    # print(c_norm_1)

    d = tf.reshape(a, (1, 2, 2, 1))
    d_norm = tf.norm(d, axis=0)
    print(d_norm)    # [[[1.] [1.]] [[1.] [1.]]] shape=(2, 2, 1)

    e = tf.Variable(c)
    with tf.GradientTape() as tape:
        e_norm_0 = e[0] / tf.norm(e[0])
        e_norm_1 = e[1] / tf.norm(e[1])

        e_norm_stacked = tf.stack([e_norm_0, e_norm_1])

    gradient = tape.gradient(e_norm_stacked, e)
    print(gradient)    # [[[[0.]] [[0.]]] [[[0.]] [[0.]]]], shape=(2, 2, 1, 1) This doesn't work.

    
def test_norm_2():
    """
    Tests `tf.math.l2_normalize`.
    """
    def _test_l2_normalize(x: tf.Variable):
        with tf.GradientTape() as tape:
            y = tf.math.l2_normalize(x, axis=0)

        print("Output:", y)
        gradient = tape.gradient(y, x)
        print("Gradient:", gradient)

    a1 = tf.constant([1, 1, 1, 1], dtype=tf.float32)
    x1 = tf.Variable(tf.reshape(a1, (2, 2, 1, 1)))
    # Output: [[[[0.70710677]] [[0.70710677]]] [[[0.70710677]] [[0.70710677]]]], shape=(2, 2, 1, 1)
    # Gradient: [[[[5.9604645e-08]] [[5.9604645e-08]]] [[[5.9604645e-08]] [[5.9604645e-08]]]], shape=(2, 2, 1, 1)
    print("\nTest 1:")
    _test_l2_normalize(x1)

    a2 = tf.constant([2, 2, 2, 2], dtype=tf.float32)
    x2 = tf.Variable(tf.reshape(a2, (2, 2, 1, 1)))
    # Output: [[[[0.70710677]] [[0.70710677]]] [[[0.70710677]] [[0.70710677]]]], shape=(2, 2, 1, 1)
    # Gradient: [[[[2.9802322e-08]] [[2.9802322e-08]]] [[[2.9802322e-08]] [[2.9802322e-08]]]], shape=(2, 2, 1, 1)
    print("\nTest 2:")
    _test_l2_normalize(x2)

    a3 = tf.constant([1, 2, 1, 2], dtype=tf.float32)
    x3 = tf.Variable(tf.reshape(a3, (2, 2, 1, 1)))
    print("\nTest 3:")
    print("Input:", x3)
    _test_l2_normalize(x3)

    a4 = tf.constant([1, 2, 1, 2], dtype=tf.float32)
    x4 = tf.Variable(tf.reshape(a4, (2, 2)))
    print("\nTest 4:")
    _test_l2_normalize(x4)


def _test_norm_2_derivatives_1():
    x = [(1.0, 1.0)]
    dy_dx1 = [((x1 ** 2) / math.sqrt(x1 ** 2 + x2 ** 2) + math.sqrt(x1 ** 2 + x2 ** 2)) / (x1 ** 2 + x2 ** 2) for x1, x2 in x]
    
    for (x1, x2), d in zip(x, dy_dx1):
        print(x1, x2, d)


def _test_norm_2_derivatives_2():
    x1 = tf.Variable(1.0)
    x2 = tf.Variable(2.0)
    with tf.GradientTape(persistent=True) as tape:
        y1 = x1 / tf.sqrt(x1 ** 2 + x2 ** 2)
        y2 = x2 / tf.sqrt(x1 ** 2 + x2 ** 2)

    print("y1:", y1)
    gradient1 = tape.gradient(y1, x1)
    print("Gradient 1:", gradient1)
    print("y2:", y2)
    gradient2 = tape.gradient(y2, x2)
    print("Gradient 2:", gradient2)


def _test_norm_2_derivatives_3():
    x = tf.Variable([1.0, 2.0])
    with tf.GradientTape() as tape:
        y = tf.math.l2_normalize(x)

    print("y:", y)
    gradient = tape.gradient(y, x)
    print("Gradient:", gradient)    # [ 0.17888546 -0.08944267] TODO: Why is the 2nd value negative?

    
# MARK: - Main
if (__name__ == "__main__"):
    # test_basics()

    # test_add_sub()

    # test_multiply_1()
    # test_multiply_2()
    # test_multiply_3()
    # test_multiply_4()

    # test_power()

    # test_softmax()

    # test_reduce_max()

    # test_argmax()

    # test_norm_1()
    # test_norm_2()
    # _test_norm_2_derivatives_1()
    _test_norm_2_derivatives_2()
    _test_norm_2_derivatives_3()
