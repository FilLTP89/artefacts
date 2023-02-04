import tensorflow as tf
import tensorflow.keras.layers as kl

from loss import (
    style_loss,
    content_loss,
    perceptual_loss,
    generator_gan_loss,
    discriminator_loss,
)


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


class PatchGAn(tf.keras.Model):
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = kl.Flatten()(x)
        x = self.dense(x)
        return x


class VGG199_feature_extractor(tf.keras.Model):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.model = tf.keras.applications.VGG19(
            weights="imagenet",
            input_shape=self.input_shape,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )


class MEDGAN(tf.keras.Model):
    """
    Implementation of the MedGAN model presented in the paper
    """

    def __init__(
        self,
        input_shape,
        learning_rate=3e-4,
        generator=None,
        discriminator=None,
        style_loss=None,
        N_g=3,
    ):
        super().__init__()

        self.shape = input_shape
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.style_loss = style_loss
        self.metrics_list = [tf.keras.metrics.RootMeanSquaredError()]
        self.N_g = N_g  # number of training iterations for generator

        self.lambda_1 = 10
        self.lambda_2 = 5
        self.lambda_3 = 1

        self.generator = generator or ConsNet(6, self.shape)
        self.discriminator = PatchGAn()
        self.feature_extractor = tf.keras.applications.VGG19(
            weights="imagenet",
            input_shape=(512, 512, 1),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

    def build_model(self):
        input = kl.Input(shape=self.shape)
        x = self.generator(input)
        output = self.discriminator(x)
        model = tf.keras.Model(input, output)
        return model

    def training_step(self, data):
        x, y = data
        for _ in range(self.N_g):
            with tf.GradientTape() as tape:
                y_pred = self.generator(x)
                y_pred_discriminator = self.discriminator(y_pred)
                y_pred_feature = self.feature_extractor(y_pred)

                y_true_discriminator = self.discriminator(y)
                y_true_feature = self.feature_extractor(y)

                # gan loss
                generator_gan_l = generator_gan_loss(y_pred_discriminator)
                # perceptual loss
                perceptual_l = perceptual_loss(y_pred_discriminator, y_true_feature)
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
            grads = tape.gradient(generator_loss, self.generator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.generator.trainable_weights)
            )

        with tf.GradientTape() as tape:
            discriminator_l = discriminator_loss(
                y_true_discriminator, y_pred_discriminator
            )
        grads = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )


if __name__ == "__main__":
    patchgan = PatchGAn()
    input = tf.random.normal((3, 512, 512, 1))
    y = patchgan(input)
    print(y.shape)
