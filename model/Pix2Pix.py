import tensorflow as tf
import tensorflow.keras.layers as kl
from metrics import ssim


class Pix2Pix(tf.keras.Model):
    def __init__(
        self,
        input_shape: int,
        nb_class: int,
        learning_rate: int = 3e-4,
    ) -> None:
        super().__init__()
        self.shape = input_shape  # Input shape has to be power of 2, 256x256 or 512x512
        self.nb_class = nb_class
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics_list = [tf.keras.metrics.RootMeanSquaredError(), ssim]

    def Generator(self):
        return

    def Discriminator(self):
        return

    def build_model(self):
        input = kl.Input(shape=self.shape)
        generator_output = self.Generator()(input)
        discriminator_output = self.Discriminator()(generator_output)
        model = tf.keras.Model(inputs=input, outputs=discriminator_output)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics_list,
        )
        return model
