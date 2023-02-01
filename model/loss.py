# TODO : Implement the loss function for the model if needed

import tensorflow as tf


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
