import tensorflow as tf
import tensorflow.keras.layers as kl

from loss import (
    style_loss,
    content_loss,
    perceptual_loss,
    generator_gan_loss,
    discriminator_loss,
)

from vgg19 import VGG19


class U_block(tf.keras.Model):

    """
    One U_block is composed of 16 convolutional layer
    Encoder part filter = [64, 128, 256, 512, 512, 512, 512, 512]
    Decoder part filter = [512, 1024, 1024, 1024, 1024, 512, 256, 128]

    Interrogation : why do the last layer 128 filters and not 1 ?
    """

    def __init__(self, shape=(512, 512, 1)) -> None:
        super().__init__()
        self.shape = shape

    def down_conv_block(self, input, filters, kernel_size=4, strides=2, padding="same"):
        x = kl.Conv2D(filters, kernel_size, strides, padding)(input)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU()(x)
        return x

    def up_conv_block(
        self,
        input,
        skip_input,
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
    ):
        x = kl.Conv2DTranspose(filters, kernel_size, strides, padding)(input)
        x = kl.BatchNormalization()(x)
        x = kl.ReLU()(x)
        x = kl.Concatenate()([x, skip_input])
        return x

    def encoder_block(self, input, filters=[64, 128, 256, 512, 512, 512, 512, 512]):
        encoder_list = []
        for i, filter in enumerate(filters):
            if i == 0:
                x = self.down_conv_block(input, filter)
                encoder_list.append(x)
            else:
                x = self.down_conv_block(encoder_list[i - 1], filter)
                encoder_list.append(x)
        return encoder_list

    def decoder_block(
        self, encoder_list, filters=[512, 1024, 1024, 1024, 1024, 512, 256, 128]
    ):
        encoder_list = encoder_list[::-1]
        y = self.up_conv_block(encoder_list[0], encoder_list[1], filters[0])
        for i, filter in enumerate(filters[2:]):
            y = self.up_conv_block(y, encoder_list[i + 2], filter)
        y = kl.Conv2DTranspose(1, 4, 2, padding="same")(y)
        return y

    def build_model(self):
        input = kl.Input(shape=self.shape)
        encoder_list = self.encoder_block(input)
        decoder_output = self.decoder_block(encoder_list)
        model = tf.keras.Model(inputs=input, outputs=decoder_output)
        return model


class ConsNet(tf.keras.Model):
    def __init__(self, n_block=6, input_shape=(512, 512, 1)) -> None:
        super().__init__()
        self.n_block = n_block
        self.shape = input_shape
        self.Ublock = [U_block().build_model() for _ in range(n_block)]

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.n_block):
            x = self.Ublock[i](x)
        return x


class PatchGAN(tf.keras.Model):
    """
    Implementation of the PatchGAN discriminator presented in the paper
    "Image-to-Image Translation with Conditional Adversarial Networks"
    and used in the MedGan model
    """

    def __init__(self, input_shape=(512, 512, 1), patch_size=70) -> None:
        super().__init__()
        self.shape = input_shape
        self.patch_size = patch_size

        self.conv1 = kl.Conv2D(64, 3, 1, "same")
        self.conv2 = kl.Conv2D(128, 3, 1, "same")
        self.batch_norm1 = kl.BatchNormalization()
        self.relu = kl.LeakyReLU()

        self.dense = kl.Dense(1, activation="sigmoid")

    def into_patch(self, input):
        bs = input.shape[0]
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Calculate the number of patches that can be extracted from the image
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]

        # Reshape the patches tensor to [num_patches, 16, 16, 1]
        patches = tf.reshape(
            patches, [bs, num_patches, self.patch_size, self.patch_size, 1]
        )
        return patches

    def call(self, input):
        x = self.into_patch(input)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_ = self.batch_norm1(x2)
        x_ = self.relu(x_)
        x_ = kl.Flatten()(x_)
        x_ = self.dense(x_)
        return ([x, x1, x2], x_)  # return features and the result of the output layer


class MEDGAN(tf.keras.Model):
    """
    Implementation of the MedGAN model presented in the paper
    """

    def __init__(
        self,
        input_shape=(512, 512, 1),
        generator=None,
        discriminator=None,
        feature_extractor=None,
        N_g=3,
    ):
        super().__init__()

        self.shape = input_shape

        self.N_g = N_g  # number of training iterations for generator

        self.lambda_1 = 10
        self.lambda_2 = 5
        self.lambda_3 = 1

        self.g_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        self.generator = generator or ConsNet(6, self.shape)
        self.discriminator = discriminator or PatchGAN()
        self.feature_extractor = feature_extractor or VGG19()

        self.style_loss_tracker = tf.keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = tf.keras.metrics.Mean(name="content_loss")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")
        self.generator_gan_loss_tracker = tf.keras.metrics.Mean(
            name="generator_gan_loss"
        )
        self.generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(
            name="discriminator_loss"
        )

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.perceptual_loss_tracker,
            self.generator_gan_loss_tracker,
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        for _ in range(
            self.N_g
        ):  # as the paper suggest it is three iterations for the generator and one for the discriminator
            with tf.GradientTape() as tape:
                y_pred = self.generator(x)

                (
                    y_pred_discriminator_features,
                    y_pred_discriminator_last_layer,
                ) = self.discriminator(
                    y_pred
                )  # get the features of the discriminator and the output of the last layer for the output of the generator

                y_pred_feature = self.feature_extractor(
                    y_pred
                )  # get the features of the feature extractor for the true samples

                y_true_feature = self.feature_extractor(y)
                (
                    y_true_discriminator_features,
                    y_true_discriminator_last_layer,
                ) = self.discriminator(
                    y
                )  # get the features of the discriminator and the output of the last layer for the true samples

                # gan loss
                generator_gan_l = generator_gan_loss(y_pred_discriminator_last_layer)
                # perceptual loss
                perceptual_l = perceptual_loss(
                    y_pred_discriminator_features, y_true_discriminator_features
                )
                # style loss
                style_l = style_loss(y_pred_feature, y_true_feature)
                # content loss
                content_l = content_loss(y_pred_feature, y_true_feature)

                generator_loss = (
                    generator_gan_l
                    + self.lambda_1 * perceptual_l
                    + self.lambda_2 * style_l
                    + self.lambda_3 * content_l
                )
            # Compute the gradiants of the loss with respect to the weights of the generator
            grads = tape.gradient(generator_loss, self.generator.trainable_weights)
            # Update the weights of the generator
            self.g_optimizer.apply_gradients(
                zip(grads, self.generator.trainable_weights)
            )
        # We create the label for the discriminator
        true_label = tf.concat(
            [
                tf.zeros_like(y_pred_discriminator_last_layer),
                tf.ones_like(y_pred_discriminator_last_layer),
            ]
        )
        y_hat = tf.concat(
            [y_pred_discriminator_last_layer, y_true_discriminator_last_layer]
        )

        with tf.GradientTape() as tape:
            discriminator_l = discriminator_loss(y_hat, true_label)
        # Compute the gradiants of the loss with respect to the weights of the discriminator
        grads = tape.gradient(discriminator_l, self.discriminator.trainable_weights)
        # Update the weights of the discriminator
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Update the loss trackers
        self.style_loss_tracker.update_state(style_l)
        self.content_loss_tracker.update_state(content_l)
        self.perceptual_loss_tracker.update_state(perceptual_l)
        self.generator_gan_loss_tracker.update_state(generator_gan_l)
        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_l)

        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "generator_gan_loss": self.generator_gan_loss_tracker.result(),
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        y_pred = self.generator(x)

        (
            y_pred_discriminator_features,
            y_pred_discriminator_last_layer,
        ) = self.discriminator(
            y_pred
        )  # get the features of the discriminator and the output of the last layer for the output of the generator

        y_pred_feature = self.feature_extractor(
            y_pred
        )  # get the features of the feature extractor for the true samples

        y_true_feature = self.feature_extractor(y)
        (
            y_true_discriminator_features,
            y_true_discriminator_last_layer,
        ) = self.discriminator(
            y
        )  # get the features of the discriminator and the output of the last layer for the true samples

        # gan loss
        generator_gan_l = generator_gan_loss(y_pred_discriminator_features)
        # perceptual loss
        perceptual_l = perceptual_loss(
            y_pred_discriminator_features, y_true_discriminator_features
        )
        # style loss
        style_l = style_loss(y_pred_feature, y_true_feature)
        # content loss
        content_l = content_loss(y_pred_feature, y_true_feature)

        generator_loss = (
            generator_gan_l
            + self.lambda_1 * perceptual_l
            + self.lambda_2 * style_l
            + self.lambda_3 * content_l
        )
        true_label = tf.concat(
            [
                tf.zeros_like(y_pred_discriminator_last_layer),
                tf.ones_like(y_pred_discriminator_last_layer),
            ]
        )
        y_hat = tf.concat(
            [y_pred_discriminator_last_layer, y_true_discriminator_last_layer]
        )
        discriminator_l = discriminator_loss(y_hat, true_label)

        return {
            "content_loss": content_l,
            "style_loss": style_l,
            "perceptual_loss": perceptual_l,
            "generator_gan_loss": generator_gan_l,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_l,
        }


def test():
    input = tf.random.normal((3, 512, 512, 1))
    input2 = tf.random.normal((3, 512, 512, 1))

    consnet = ConsNet(6, (512, 512, 1))
    generator_output = consnet(input)

    vgg19 = VGG19()
    feature_extractor_pred_features, feature_extractor_pred__last_layer = vgg19(
        generator_output
    )
    feature_extrator_true_features, feature_extractor_true_last_layer = vgg19(input2)

    patchgan = PatchGAN()
    discriminato_pred_output, discriminator_pred_last_layer = patchgan(generator_output)
    discriminator_true_output, discriminator_true_last_layer = patchgan(input2)

    style_loss_ = style_loss(
        feature_extractor_pred_features, feature_extrator_true_features
    )
    content_loss_ = content_loss(
        feature_extractor_pred_features, feature_extrator_true_features
    )
    perceptual_loss_ = perceptual_loss(
        discriminato_pred_output, discriminator_true_output
    )
    generator_gan_loss_ = generator_gan_loss(discriminator_pred_last_layer)
    discriminator_loss_ = discriminator_loss(
        discriminator_true_last_layer, discriminator_pred_last_layer
    )
    print("style_loss", style_loss_)
    print("content_loss", content_loss_)
    print("perceptual_loss", perceptual_loss_)
    print("generator_gan_loss", generator_gan_loss_)
    print("discriminator_loss", discriminator_loss_)


def patchgan_test():
    patchgan = PatchGAN()
    x = tf.random.normal((3, 512, 512, 1))
    y = tf.random.normal((3, 512, 512, 1))
    y_hat_feature, y_hat_lastlayer = patchgan(x)
    y_feature, y_last_layer = patchgan(y)
    print(perceptual_loss(y_hat_feature, y_feature))
    print(generator_gan_loss(y_hat_lastlayer))


def vgg19_test():
    vgg19 = VGG19()
    x = tf.random.normal((3, 512, 512, 1))
    y = tf.random.normal((3, 512, 512, 1))
    y_hat_feature = vgg19(x)
    y_feature = vgg19(y)
    print(style_loss(y_hat_feature, y_feature))
    print(content_loss(y_hat_feature, y_feature))


def consnet_test():
    consnet = ConsNet(6, (512, 512, 1))
    x = tf.random.normal((3, 512, 512, 1))
    y_hat = consnet(x)
    print(y_hat.shape)


def medgann_test():
    medgan = MEDGAN()
    x = tf.random.normal((3, 512, 512, 1))
    y = tf.random.normal((3, 512, 512, 1))
    medgan.compile()
    medgan.fit(x, y, epochs=1)


if __name__ == "__main__":
    medgann_test()
