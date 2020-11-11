import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
from tensorflow import keras


# MARK: - Basics
def test_scalar():
    x = tf.Variable(3.0)
    # print(x)

    with tf.GradientTape() as tape:
        y = x ** 2

    dy_dx = tape.gradient(y, x)
    print(dy_dx)    # tf.Tensor(6.0, shape=(), dtype=float32)


def test_watch():
    """
    By default, all trainable variables are watched.

    Regular `Tensor`s are not watched by default.

    Use `tape.watch` to watch it.
    """
    x1 = tf.constant(3.0, name="x1")
    x2 = tf.Variable(3.0, trainable=False, name="x2")
    x3 = tf.Variable(3.0, name="x3")    # Names appear in `watched_variables()`.
    x4 = tf.Variable(3.0, name="x4")

    with tf.GradientTape() as tape:
        # tape.watch(x1)
        y = x1 ** 2 + x2 ** 2 + x3 ** 2

    [dy_dx1, dy_dx2, dy_dx3, dy_dx4] = tape.gradient(y, [x1, x2, x3, x4])
    print(f"dy_dx1: {dy_dx1}")    # None
    print(f"dy_dx2: {dy_dx2}")    # None
    print(f"dy_dx3: {dy_dx3}")    # 6.0
    print(f"dy_dx4: {dy_dx4}")    # None (because `x4` wasn't used)

    print(f"Watched variables: {tape.watched_variables()}")    # Only `x3` appears here. Constants do not appear here (even manually watched).


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


# MARK: - Model
def test_model():
    layer = keras.layers.Dense(2, activation="relu")
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:
        # Forward pass
        y = layer(x)
        loss = tf.reduce_mean(y**2)

    # Calculate gradients with respect to every trainable variable
    grad = tape.gradient(loss, layer.trainable_variables)

    for var, g in zip(layer.trainable_variables, grad):
        # dense/kernel:0, shape: (3, 2)
        # dense/bias:0, shape: (2,)
        print(f"{var.name}, shape: {g.shape}")


# MARK: - Persistence
def test_persistence_1():
    """
    `tape.gradient` can only be called once on a non-persistent tape.
    """
    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        y = x ** 2
        z = y ** 2

    dy_dx = tape.gradient(y, x)
    dz_dy = tape.gradient(z, y)

    print(f"dy_dx: {dy_dx}")    # 6.0
    print(f"dz_dy: {dz_dy}")    # Should be 18.0, but received "RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes." instead.


def test_persistence_2():
    x = tf.Variable(3.0)
    with tf.GradientTape(persistent=True) as tape:
        y = x ** 2
        z = y ** 2

    dy_dx = tape.gradient(y, x)
    dz_dy = tape.gradient(z, y)

    print(f"dy_dx: {dy_dx}")    # 6.0
    print(f"dz_dy: {dz_dy}")    # 18.0


# MARK: - Main
if (__name__ == "__main__"):
    # test_scalar()
    # test_watch()
    # test_multi_axes()
    # test_multi_variables()

    # test_model()

    # test_persistence_1()
    test_persistence_2()
