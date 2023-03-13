from ops import *
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.metrics as km
import tensorflow.keras.constraints as kc
import tensorflow.keras.initializers as ki

tfd = tfp.distributions

loss_names = ["AdvDlossX", "AdvGlossX"]


class DeepMAR(tf.keras.Model):
    def __init__(self, options, name="DeepMAR"):
        """

        Args:
            sess: TensorFlow session
            batchSize: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            nFiltersGenerator: (optional) Dimension of gen filters in first conv layer. [64]
            nFiltersDiscriminator: (optional) Dimension of discrim filters in first conv layer. [64]
            nXchannels: (optional) Dimension of input image. For grayscale input, set to 1.
            output_c_dim: (optional) Dimension of output image. For grayscale input, set to 1.
        """
        super(DeepMAR, self).__init__(name=name)
        """
            Setup
        """
        self.__dict__.update(options)

        # self.sess = sess
        self.is_grayscale = self.nXchannels == 1

        # self.checkpoint_dir = checkpoint_dir
        self.keep_prob = 1.0

        self.loss_trackers = {
            "{:>s}_tracker".format(l): km.Mean(name=l) for l in loss_names
        }
        self.loss_val = {"{:>s}".format(l): 0.0 for l in loss_names}

        self.BuildModels()

    def reset_metrics(self):
        for k, v in self.loss_trackers.items():
            v.reset_states()

    def get_config(self):
        config = super().get_config().copy()
        config.update({"size": self.Xshape})
        return config

    @property
    def metrics(self):
        return list(self.loss_trackers.values())

    def BuildModels(self):

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="d_bn1")
        self.d_bn2 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="d_bn2")
        self.d_bn3 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="d_bn3")
        self.d_bn4 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="d_bn4")
        self.d_bn5 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="d_bn5")
        self.d_bn6 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="d_bn6")

        self.g_bn_e2 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_e2")
        self.g_bn_e3 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_e3")
        self.g_bn_e4 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_e4")
        self.g_bn_e5 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_e5")
        self.g_bn_e6 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_e6")

        self.g_bn_d3 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_d3")
        self.g_bn_d4 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_d4")
        self.g_bn_d5 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_d5")
        self.g_bn_d6 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_d6")
        self.g_bn_d7 = kl.BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_bn_d7")

        self.G_X = self.BuildGX()
        self.D_X = self.BuildDX()

        self.models = [self.G_X, self.D_X]

    def compile(self, optimizers, losses, **kwargs):

        super(CGAN_model, self).compile(**kwargs)
        """
            Optimizers
        """
        self.__dict__.update(optimizers)
        """
            Losses
        """
        self.__dict__.update(losses)

    @tf.function
    def train_step(self, XC):
        if isinstance(XC, tuple):
            X, metadata = XC

        self.batchSize = tf.shape(X)[0]

        # Train discriminative part
        for _ in range(self.nCritic):

            # Tape gradients
            with tf.GradientTape(persistent=True) as tape:

                # Encode real signals X
                X_hat = self.G_X(X, training=True)

                # Discriminates real (X,z_hat) from false (X_hat,z_hat)
                D_X_real = self.D_X(X, training=True)
                D_X_fake = self.D_X(X_hat, training=True)

                # Compute Z adversarial loss (JS(x,z))
                AdvDlossXz = self.AdvDlossX(D_X_real, D_X_fake)

        # Compute our own metrics
        for k, v in self.loss_trackers.items():
            v.update_state(self.loss_val[k.strip("_tracker")])

        return {k: v.result() for v in self.loss_trackers.values()}

    def BuildGX(self):
        X = kl.Input(
            shape=(self.imageWidth, self.imageHeight, self.nXchannels),
            dtype=tf.float32,
            name="incomplete_sinograms",
        )
        Xp = kl.ZeroPadding2D(padding=(24, 24))(X)
        self.e1 = kl.Conv2D(
            filters=self.nFiltersGenerator,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_e1_conv",
        )(Xp)
        e1a = kl.LeakyReLU(alpha=0.2)(self.e1)
        self.e2 = kl.Conv2D(
            filters=self.nFiltersGenerator * 2,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_e2_conv",
        )(e1a)
        e2 = self.g_bn_e2(self.e2)
        e2a = kl.LeakyReLU(alpha=0.2)(e2)
        self.e3 = kl.Conv2D(
            filters=self.nFiltersGenerator * 4,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_e3_conv",
        )(e2a)
        e3 = self.g_bn_e3(self.e3)
        e3a = kl.LeakyReLU(alpha=0.2)(e3)
        self.e4 = kl.Conv2D(
            filters=self.nFiltersGenerator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_e4_conv",
        )(e3a)
        e4 = self.g_bn_e4(self.e4)
        e4a = kl.LeakyReLU(alpha=0.2)(e4)
        self.e5 = kl.Conv2D(
            filters=self.nFiltersGenerator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_e5_conv",
        )(e4a)
        e5 = self.g_bn_e5(self.e5)
        e5a = kl.LeakyReLU(alpha=0.2)(e5)
        self.e6 = kl.Conv2D(
            filters=self.nFiltersGenerator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_e6_conv",
        )(e5a)
        e6 = self.g_bn_e6(self.e6)
        e6a = kl.Activation("relu")(e6)

        self.d3 = kl.Conv2DTranspose(
            filters=self.nFiltersGenerator * 8,
            kernel_size=(self.kh, self.kw),
            input_shape=tuple(e6a.shape[1:]),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_d3",
        )(e6a)
        d3 = self.g_bn_d3(self.d3)
        import pdb

        pdb.set_trace()
        d3 = tf.concat([d3, e5], -1)
        d3a = kl.Activation("relu")(d3)
        self.d4 = kl.Conv2DTranspose(
            filters=self.nFiltersGenerator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_d4",
        )(d3a)
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat([d4, e4], 3)
        d4a = kl.Activation("relu")(d4)

        self.d5 = kl.Conv2DTranspose(
            filters=self.nFiltersGenerator * 4,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_d5",
        )(d4a)
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat([d5, e3], 3)
        d5a = kl.Activation("relu")(d5)

        self.d6 = kl.Conv2DTranspose(
            filters=self.nFiltersGenerator * 2,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_d6",
        )(d5a)
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat([d6, e2], 3)
        d6a = kl.Activation("relu")(d6)

        self.d7 = kl.Conv2DTranspose(
            filters=self.nFiltersGenerator,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_d7",
        )(d6a)
        d7 = self.g_bn_d7(self.d7)
        d7 = tf.concat([d7, self.e1], 3)
        d7 = kl.Dropout(noise_shape=(self.batchSize, 1, 1, 2 * self.nFiltersGenerator))(
            d7
        )
        d7a = kl.Activation("relu")(d7)

        Xhat = kl.Conv2DTranspose(
            filters=self.output_c_dim,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="g_d8",
        )(d7a)

        G_X = tf.keras.Model(inputs=X, outputs=Xhat, name="GX")
        return G_X

    def BuildDX(self):

        X = kl.Input(
            shape=(self.batchSize, 1024, 768, self.nXchannels),
            dtype=tf.float32,
            name="incomplete_sinograms",
        )
        # image,
        # y=None,
        # reuse=False,
        # ):
        # resizing images for discriminator
        self.image = tf.image.resize_images(X, [512, 384])
        self.h0 = kl.Conv2D(
            filters=self.nFiltersDiscriminator,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh0_conv",
        )(self.image)
        h0 = kl.LeakyReLU(alpha=0.2)(self.h0)
        self.h1 = kl.Conv2D(
            filters=self.nFiltersDiscriminator * 2,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh1_conv",
        )(h0)
        h1 = kl.LeakyReLU(alpha=0.2)(self.d_bn1(self.h1))
        self.h2 = kl.Conv2D(
            filters=self.nFiltersDiscriminator * 4,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh2_conv",
        )(h1)
        h2 = kl.LeakyReLU(alpha=0.2)(self.d_bn2(self.h2))
        self.h3 = kl.Conv2D(
            filters=self.nFiltersDiscriminator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh3_conv",
        )(h2)
        h3 = kl.LeakyReLU(alpha=0.2)(self.d_bn3(self.h3))
        self.h4 = kl.Conv2D(
            filters=self.nFiltersDiscriminator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh4_conv",
        )(h3)
        h4 = kl.LeakyReLU(alpha=0.2)(self.d_bn4(self.h4))
        self.h5 = kl.Conv2D(
            filters=self.nFiltersDiscriminator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh5_conv",
        )(h4)
        h5 = kl.LeakyReLU(alpha=0.2)(self.d_bn5(self.h5))
        self.h6 = kl.Conv2D(
            filters=self.nFiltersDiscriminator * 8,
            kernel_size=(self.kh, self.kw),
            strides=(self.dh, self.dw),
            padding="same",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh6_conv",
        )(h5)
        h6 = kl.LeakyReLU(alpha=0.2)(self.d_bn6(self.h6))
        h7f = kl.Flatten(h6)
        p_X = kl.Dense(
            activation="sigmoid",
            kernel_initializer=ki.TruncatedNormal(mean=0.0, stddev=self.stddev),
            bias_initializer="zeros",
            name="dh7_lin",
        )(h7f)

        D_X = tf.keras.Model(inputs=X, outputs=p_X, name="DX")
        return D_X


if __name__ == "__main__":
    model = DeepMAR().BuildModels()
    model.compile()
