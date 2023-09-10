import tensorflow as tf
from keras import backend as K

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



def iou(y_true, y_pred, smooth=1.):
    y_pred = K.round(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def accuracy(y_true, y_pred):
    y_pred = K.round(y_pred)
    return K.mean(K.equal(y_true, y_pred))

def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
