import tensorflow as tf


def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)).numpy()


def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=2.0)).numpy()


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def accuracy(y_true, y_pred):
    """
    Compute accuracy for segmentation.
    """
    y_true_f = tf.cast(tf.round(y_true), tf.float32)
    y_pred_f = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(y_pred_f, y_true_f), tf.float32)
    return tf.reduce_mean(correct_predictions)

def precision(y_true, y_pred):
    """
    Compute precision for segmentation.
    """
    y_true_f = tf.cast(tf.round(y_true), tf.float32)
    y_pred_f = tf.round(y_pred)
    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def recall(y_true, y_pred):
    """
    Compute recall for segmentation.
    """
    y_true_f = tf.cast(tf.round(y_true), tf.float32)
    y_pred_f = tf.round(y_pred)
    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)
    return true_positives / (actual_positives + tf.keras.backend.epsilon())

def f1_score(y_true, y_pred):
    """
    Compute F1-score for segmentation.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r )