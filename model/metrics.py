import tensorflow as tf


def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=2.0)


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def nmi(y_true, y_pred):
    return tf.reduce_mean(tf.image.normalized_mutual_information(y_true, y_pred, 2))
