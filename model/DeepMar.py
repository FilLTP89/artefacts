import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl


class Generator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        # a
        p1 = kl.Conv2D(64, (5, 5), padding="same", strides=2)(inputs)
        p1 = kl.LeakyReLU()(p1)

        # b
        p2 = kl.Conv2D(128, (5, 5), padding="same", strides=2)(p1)
        p2 = kl.BatchNormalization()(p2)
        p2 = kl.LeakyReLU()(p2)

        # b
        p3 = kl.Conv2D(256, (5, 5), padding="same", strides=2)(p2)
        p3 = kl.BatchNormalization()(p3)
        p3 = kl.LeakyReLU()(p3)

        # b
        p4 = kl.Conv2D(512, (5, 5), padding="same", strides=2)(p3)
        p4 = kl.BatchNormalization()(p4)
        p4 = kl.LeakyReLU()(p4)

        # b
        p5 = kl.Conv2D(512, (5, 5), padding="same", strides=2)(p4)
        p5 = kl.BatchNormalization()(p5)
        p5 = kl.LeakyReLU()(p5)

        # b
        p6 = kl.Conv2D(512, (5, 5), padding="same", strides=2)(p5)
        p6 = kl.BatchNormalization()(p6)
        p6 = kl.LeakyReLU()(p6)

        # c
        p6 = kl.Conv2DTranspose(512, (5, 5), padding="same", strides=2)(p6)
        p6 = kl.BatchNormalization()(p6)
        p6 = kl.LeakyReLU()(p6)
        p6 = kl.Concatenate()([p6, p5])

        # c
        p6 = kl.Conv2DTranspose(512, (5, 5), padding="same", strides=2)(p6)
        p6 = kl.BatchNormalization()(p6)
        p6 = kl.LeakyReLU()(p6)
        p6 = kl.Concatenate()([p6, p4])

        # c
        p6 = kl.Conv2DTranspose(256, (5, 5), padding="same", strides=2)(p6)
        p6 = kl.BatchNormalization()(p6)
        p6 = kl.LeakyReLU()(p6)
        p6 = kl.Concatenate()([p6, p3])

        # c
        p6 = kl.Conv2DTranspose(128, (5, 5), padding="same", strides=2)(p6)
        p6 = kl.BatchNormalization()(p6)
        p6 = kl.LeakyReLU()(p6)
        p6 = kl.Concatenate()([p6, p2])

        # c
        p6 = kl.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(p6)
        p6 = kl.BatchNormalization()(p6)
        p6 = kl.LeakyReLU()(p6)
        p6 = kl.Concatenate()([p6, p1])

        # d
        x = kl.Conv2DTranspose(1, (5, 5), padding="same", strides=2)(p6)
        x = kl.Dropout(0.2)(x)
        return x


class Discriminator(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):

        # a
        input = kl.Conv2D(64, (5, 5), padding="same", strides=2)(inputs)
        input = kl.LeakyReLU()(input)

        # b
        input = kl.Conv2D(128, (5, 5), padding="same", strides=2)(input)
        input = kl.BatchNormalization()(input)
        input = kl.LeakyReLU()(input)

        # b
        input = kl.Conv2D(256, (5, 5), padding="same", strides=2)(input)
        input = kl.BatchNormalization()(input)
        input = kl.LeakyReLU()(input)

        # b
        input = kl.Conv2D(512, (5, 5), padding="same", strides=2)(input)
        input = kl.BatchNormalization()(input)
        input = kl.LeakyReLU()(input)

        # b
        input = kl.Conv2D(512, (5, 5), padding="same", strides=2)(input)
        input = kl.BatchNormalization()(input)
        input = kl.LeakyReLU()(input)

        # b
        input = kl.Conv2D(512, (5, 5), padding="same", strides=2)(input)
        input = kl.BatchNormalization()(input)
        input = kl.LeakyReLU()(input)

        # b
        input = kl.Conv2D(512, (5, 5), padding="same", strides=2)(input)
        input = kl.BatchNormalization()(input)
        input = kl.LeakyReLU()(input)

        input = kl.Flatten()(input)
        output = kl.Dense(1, activation="sigmoid")(input)

        return output


class DeepMar(tf.keras.Model):
    def __init__(self, learning_rate=3e-4):
        super().__init__()

        self.learning_rate = learning_rate
        self.shape = (512, 512, 1)

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.g_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.d_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")

        self.gen_adv_loss_tracker = tf.keras.metrics.Mean(name="gen_adv_loss")
        self.gen_l2_loss_tracker = tf.keras.metrics.Mean(name="gen_l2_loss")

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.gen_loss_tracker,
            self.gen_adv_loss_tracker,
            self.gen_l2_loss_tracker,
        ]

    def training_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            fake_y = self.generator(x)
            disc_fake_output = self.discriminator(tf.concat([x, fake_y], axis=-1))
            disc_real_output = self.discriminator(tf.concat[x, y], axis=-1)

            zeros_output = tf.zeros_like(disc_fake_output)
            ones_output = tf.ones_like(disc_real_output)
            output = tf.concat([zeros_output, ones_output], axis=0)

            disc_loss = tf.keras.losses.binary_crossentropy(
                output, tf.concat([disc_fake_output, disc_real_output], axis=0)
            )

            gen_adv_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(disc_fake_output), disc_fake_output
            )
            gen_l2_loss = tf.keras.losses.mean_squared_error(y, fake_y)
            gen_loss = gen_adv_loss + gen_l2_loss

        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        self.d_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        self.disc_loss_tracker.update_state(disc_loss)
        self.gen_loss_tracker.update_state(gen_loss)
        self.gen_adv_loss_tracker.update_state(gen_adv_loss)
        self.gen_l2_loss_tracker.update_state(gen_l2_loss)

        return {
            "gen_adv_loss": self.gen_adv_loss_tracker.result(),
            "gen_l2_loss": self.gen_l2_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        fake_y = self.generator(x)
        disc_fake_output = self.discriminator(tf.concat([x, fake_y], axis=-1))
        disc_real_output = self.discriminator(tf.concat[x, y], axis=-1)

        zeros_output = tf.zeros_like(disc_fake_output)
        ones_output = tf.ones_like(disc_real_output)

        output = tf.concat([zeros_output, ones_output], axis=0)
        disc_loss = tf.keras.losses.binary_crossentropy(
            output, tf.concat([disc_fake_output, disc_real_output], axis=0)
        )

        gen_adv_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(disc_fake_output), disc_fake_output
        )
        gen_l2_loss = tf.keras.losses.mean_squared_error(y, fake_y)
        gen_loss = gen_adv_loss + gen_l2_loss


        self.disc_loss_tracker.update_state(disc_loss)
        self.gen_loss_tracker.update_state(gen_loss)
        self.gen_adv_loss_tracker.update_state(gen_adv_loss)
        self.gen_l2_loss_tracker.update_state(gen_l2_loss)

        return {
            "gen_adv_loss": self.gen_adv_loss_tracker.result(),
            "gen_l2_loss": self.gen_l2_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result(),
        }

    def call(self, x):
        return self.generator(x)


if __name__ == "__main__":
    deepmar = DeepMar()
    x = tf.random.normal((1, 512, 512, 1))
    x1 = tf.random.normal((1, 512, 251284, 1))
    y = deepmar(x)
    y2 = deepmar.discriminator(tf.concat([x, y], axis=-1))
    print(y.shape, y2.shape)
