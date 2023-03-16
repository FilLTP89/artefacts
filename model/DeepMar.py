import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras import Sequential


class Generator(tf.keras.Model):
    def __init__(self, shape=(512, 512, 1), **kwargs):
        super().__init__()
        self.shape = shape

        self.a1 = Sequential(
            [
                kl.Conv2D(64, (5, 5), padding="same", strides=2),
                kl.LeakyReLU(),
            ]
        )

        self.b1 = Sequential(
            [
                kl.Conv2D(128, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b2 = Sequential(
            [
                kl.Conv2D(256, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b3 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b4 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b5 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.c1 = Sequential(
            [
                kl.Conv2DTranspose(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.c2 = Sequential(
            [
                kl.Conv2DTranspose(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.c3 = Sequential(
            [
                kl.Conv2DTranspose(256, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.c4 = Sequential(
            [
                kl.Conv2DTranspose(128, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.c5 = Sequential(
            [
                kl.Conv2DTranspose(64, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.d1 = Sequential(
            [
                kl.Conv2DTranspose(1, (5, 5), padding="same", strides=2),
                kl.Dropout(0.2),
                kl.Activation("sigmoid"),
            ]
        )

    def call(self, inputs, training=False):

        p1 = self.a1(inputs)
        p2 = self.b1(p1)
        p3 = self.b2(p2)
        p4 = self.b3(p3)
        p5 = self.b4(p4)
        p6 = self.b5(p5)

        p6 = self.c1(p6)
        p6 = kl.Concatenate()([p6, p5])
        p6 = self.c2(p6)
        p6 = kl.Concatenate()([p6, p4])
        p6 = self.c3(p6)
        p6 = kl.Concatenate()([p6, p3])
        p6 = self.c4(p6)
        p6 = kl.Concatenate()([p6, p2])
        p6 = self.c5(p6)
        p6 = kl.Concatenate()([p6, p1])

        p6 = self.d1(p6)
        return p6


class Discriminator(tf.keras.Model):
    def __init__(self, shape=(512, 512, 1), **kwargs):
        super().__init__()
        self.shape = shape

        self.a1 = Sequential(
            [
                kl.Conv2D(64, (5, 5), padding="same", strides=2),
                kl.LeakyReLU(),
            ]
        )

        self.b1 = Sequential(
            [
                kl.Conv2D(128, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b2 = Sequential(
            [
                kl.Conv2D(256, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b3 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b4 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b5 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.b6 = Sequential(
            [
                kl.Conv2D(512, (5, 5), padding="same", strides=2),
                kl.BatchNormalization(),
                kl.LeakyReLU(),
            ]
        )

        self.c1 = Sequential([kl.Flatten(), kl.Dense(1, activation="sigmoid")])

    def call(
        self,
        inputs,
    ):

        x = self.a1(inputs)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)

        output = self.c1(x)
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

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as grad_tape:
            fake_y = self.generator(x)
            disc_fake_output = self.discriminator(tf.concat([x, fake_y], axis=-1))

            gen_adv_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(disc_fake_output), disc_fake_output
            )
            gen_l2_loss = tf.keras.losses.mean_squared_error(y, fake_y)
            gen_loss = gen_adv_loss + gen_l2_loss

        gen_grads = grad_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        with tf.GradientTape() as disc_tape:
            disc_fake_output = self.discriminator(tf.concat([x, fake_y], axis=-1))
            disc_real_output = self.discriminator(tf.concat([x, y], axis=-1))
            output = tf.concat(
                [tf.zeros_like(disc_fake_output), tf.ones_like(disc_real_output)],
                axis=0,
            )

            disc_loss = tf.keras.losses.binary_crossentropy(
                output, tf.concat([disc_fake_output, disc_real_output], axis=0)
            )
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
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
        disc_real_output = self.discriminator(tf.concat([x, y], axis=-1))

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
