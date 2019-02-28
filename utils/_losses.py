import tensorflow as tf

""" LOSSES """


def minkowsky_loss(x, y, p=1.):
    assert p >= 1., "Minkowsky Distance must be computed on p >= 1"

    p = float(p)

    if p == 1.:
        return tf.losses.absolute_difference(x, y, reduction=tf.losses.Reduction.NONE)
    else:
        per_sample_loss = tf.pow(tf.reduce_sum(tf.pow(tf.abs(x - y), p), axis=-1), 1. / p)
        return per_sample_loss


def _gram_matrix(x):
    shape = [x.shape[0], tf.reduce_prod(x.shape[1:])]
    features = tf.reshape(x, shape)
    features = features - 1.

    gram = tf.matmul(features, tf.transpose(features))
    return gram


def perceptual_loss(x, y):
    Gx = _gram_matrix(x)
    Gy = _gram_matrix(y)

    x_norm = tf.square(tf.reduce_prod(x.shape[1:]))
    y_norm = tf.square(tf.reduce_prod(y.shape[1:]))

    loss = tf.reduce_sum(tf.square(Gx - Gy)) / tf.cast(x_norm * y_norm, tf.float32)
    loss = tf.reshape(loss, [1, 1])
    return loss


def variation_loss(x1, x2, k=1):
    if len(x1.shape) == 3:  # Time series
        a = x1[:, :-k, :] - x2[:, k:, :]
        b = x2[:, :-k, :] - x2[:, k:, :]
        return tf.reduce_mean(tf.pow(tf.abs(a + b), 1.25), axis=-1)
    else:  # Images
        a = x1[:, :-1, :-1, :] - x1[:, 1:, 1:, :]
        b = x2[:, :-1, :-1, :] - x2[:, 1:, 1:, :]
        return tf.reduce_mean(tf.pow(tf.abs(a + b), 1.25), axis=-1)


# def variation_loss(x1, x2, k=1):
#     if len(x1.shape) == 3:  # Time series
#         a = x1[:, :-k, :] - x1[:, k:, :]
#         b = x2[:, :-k, :] - x2[:, k:, :]
#         return tf.reduce_mean(tf.pow(tf.abs(a + b), 1.25), axis=-1)
#     else:  # Images
#         a = x1[:, :-1, :-1, :] - x1[:, 1:, 1:, :]
#         b = x2[:, :-1, :-1, :] - x2[:, 1:, 1:, :]
#         return tf.reduce_mean(tf.pow(tf.abs(a + b), 1.25), axis=-1)


def smoothing_loss(x, width):
    if len(x.shape) == 3:  # Time series
        kernel = tf.ones([width, x.shape[-1], x.shape[-1]])
        kernel = kernel / tf.cast(width, tf.float32)

        output = tf.nn.conv1d(x, kernel, stride=1, padding='SAME')

        return tf.losses.mean_squared_error(x, output, reduction=tf.losses.Reduction.NONE)
    else:  # Images
        kernel = tf.ones([width, width, x.shape[-1], x.shape[-1]])
        kernel = kernel / tf.cast(width * width, tf.float32)

        output = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

        return tf.losses.mean_squared_error(x, output, reduction=tf.losses.Reduction.NONE)

