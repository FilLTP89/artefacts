import tensorflow as tf
import tensorflow.keras.layers as kl


class PatchGAN(tf.keras.Model):
    """
    Implementation of the PatchGAN discriminator presented in the paper
    "Image-to-Image Translation with Conditional Adversarial Networks"
    and used in the MedGan model
    """

    def __init__(self, input_shape=(512, 512, 1)) -> None:
        super().__init__()
        self.shape = input_shape

        self.block_1 = tf.keras.Sequential(
            [
                kl.Conv2D(64, 4, 2, padding="same"),
                kl.LeakyReLU(),
            ]
        )
        self.block_2 = tf.keras.Sequential(
            [
                kl.Conv2D(128, 4, 2, padding="same"),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )
        self.block_3 = tf.keras.Sequential(
            [
                kl.Conv2D(256, 4, 2, padding="same"),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )
        self.block_4 = tf.keras.Sequential(
            [
                kl.Conv2D(512, 4, 2, padding="same"),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.dense = kl.Dense(1, activation="sigmoid")

    def call(self, input):
        x = input
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        x_ = self.dense(x4)
        return (
            [x, x1, x2, x3, x4],
            x_,
        )  # return features and the result of the output layer


if __name__ == "__main__":
    patchgan = PatchGAN((256, 256, 1))
    x = tf.random.normal((3, 256, 256, 1))
    y_hat, _ = patchgan(x)
    for block in y_hat:
        print(block.shape)
