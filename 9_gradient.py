import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_scalar():
    x = tf.Variable(3.0)
    # print(x)

    with tf.GradientTape() as tape:
        y = x ** 2

    dy_dx = tape.gradient(y, x)
    print(dy_dx)    # tf.Tensor(6.0, shape=(), dtype=float32)


def test_constant():
    """
    May only calculate gradient for a `Variable`.
    """
    x = tf.constant(3.0)

    with tf.GradientTape() as tape:
        y = x ** 2

    dy_dx = tape.gradient(y, x)
    print(dy_dx)    # None


def test_multi_axes():
    x = tf.Variable(tf.reshape(tf.range(6, dtype=tf.float32), (2, 3)))

    with tf.GradientTape() as tape:
        y = x ** 2

    dy_dx = tape.gradient(y, x)
    print(dy_dx)    # tf.Tensor([[ 0.  2.  4.] [ 6.  8. 10.]], shape=(2, 3), dtype=float32)


def test_multi_variables():
    w = tf.Variable(tf.random.normal((3, 2)), name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]

    with tf.GradientTape(persistent=True) as tape:
        y = x @ w + b
        loss = tf.reduce_mean(y**2)

    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print(dl_dw)    # tf.Tensor([[-0.41205198 -1.9556504 ] [-0.82410395 -3.911301  ] [-1.236156   -5.8669515 ]], shape=(3, 2), dtype=float32)
    print(dl_db)    # tf.Tensor([-0.41205198 -1.9556504 ], shape=(2,), dtype=float32)


if (__name__ == "__main__"):
    # test_scalar()
    # test_constant()
    test_multi_axes()
    test_multi_variables()
