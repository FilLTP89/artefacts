import tensorflow as tf
import tensorflow.keras.layers as kl
from metrics import ssim


"""
Baseline : Return input as output
"""


class Baseline(tf.keras.Model):
    def __init__(self, input_shape, nb_class, learning_rate=3e-4):
        super().__init__()
        self.shape = input_shape  # Input shape has to be power of 2, 256x256 or 512x512
        self.nb_class = nb_class
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics_list = [tf.keras.metrics.RootMeanSquaredError(), ssim]

    def build_model(self):
        inputs = kl.Input(shape=self.shape)
        output = inputs
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics_list
        )
        return model
