import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K


from vgg19 import VGG19
from PatchGAN import PatchGAN
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
        self.Ublock = [U_block(self.shape).build_model() for _ in range(n_block)]

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.n_block):
            x = self.Ublock[i](x)
        return x



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
        learning_rate = 3e-5,
        N_g=3,
    ):
        super().__init__()

        self.shape = input_shape

        self.N_g = N_g  # number of training iterations for generator
        # TO DO : Do training as 1-1 and change the learning rate of the generator to be higher ?

        self.lambda_1 = 20
        self.lambda_2 = 1e-4
        self.lambda_3 = 1e-4

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate, 0.5, clipvalue = 5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate, 0.5, clipvalue = 5)

        self.generator = generator or ConsNet(6, self.shape)
        self.discriminator = discriminator or PatchGAN(self.shape)
        self.feature_extractor = feature_extractor or VGG19(self.shape)

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
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.style_loss_tracker, self.content_loss_tracker,
         self.perceptual_loss_tracker, self.generator_gan_loss_tracker, self.generator_loss_tracker, self.discriminator_loss_tracker]

    def train_step(self, data):
        x, y = data
        for _ in range(self.N_g):  # N_g = 3
            with tf.GradientTape() as gen_tape:
                y_pred = self.generator(x)

                (y_pred_discriminator_features,y_pred_discriminator_last_layer) = self.discriminator(y_pred)
                  # get the features of the discriminator and the output of the last layer for the output of the generator

                y_pred_feature = self.feature_extractor(y_pred)  
                # get the features of the feature extractor for the true samples
                y_true_feature = self.feature_extractor(y)
                (y_true_discriminator_features,_) = self.discriminator(y)  
                # get the features of the discriminator and the output of the last layer for the true samples
                # gan loss
                generator_gan_l = generator_gan_loss(y_pred_discriminator_last_layer)
                # perceptual loss
                perceptual_l = perceptual_loss(y_true_discriminator_features,y_pred_discriminator_features)
                # style loss
                style_l = style_loss(y_pred_feature, y_true_feature)
                # content loss
                content_l = content_loss(y_pred_feature, y_true_feature)

                generator_loss = (generator_gan_l
                    + self.lambda_1 * perceptual_l
                    + self.lambda_2 * style_l
                    + self.lambda_3 * content_l
                )
            # Compute the gradiants of the loss with respect to the weights of the generator
            gen_grads = gen_tape.gradient(generator_loss, self.generator.trainable_weights)
            # Update the weights of the generator
            self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))

        with tf.GradientTape() as disc_tape:
            _, y_pred_discriminator = self.discriminator(y_pred)
            _, y_true_discriminator = self.discriminator(y)
            y_hat = tf.concat([y_pred_discriminator, y_true_discriminator], axis=0)
            true_label = tf.concat([tf.zeros_like(y_pred_discriminator),tf.ones_like(y_true_discriminator)],axis=0)
            # Discriminator loss
            discriminator_l = discriminator_loss(true_label, y_hat)

            # Compute the gradiants of the loss with respect to the weights of the discriminator
        disc_grads = disc_tape.gradient(discriminator_l, self.discriminator.trainable_weights)

        # Update the weights of the discriminator
        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

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
        (y_pred_discriminator_features,y_pred_discriminator_last_layer) = self.discriminator(y_pred)  
        # get the features of the discriminator and the output of the last layer for the output of the generator
        y_pred_feature = self.feature_extractor(y_pred)  # get the features of the feature extractor for the true samples
        y_true_feature = self.feature_extractor(y)
        (y_true_discriminator_features,y_true_discriminator_last_layer) = self.discriminator(y)  
        # get the features of the discriminator and the output of the last layer for the true samples

        # gan loss
        generator_gan_l = generator_gan_loss(y_pred_discriminator_last_layer)
        # perceptual loss
        perceptual_l = perceptual_loss(y_true_discriminator_features,y_pred_discriminator_features)
        # style loss
        style_l = style_loss(y_true_feature,y_pred_feature)
        # content loss
        content_l = content_loss(y_true_feature,y_pred_feature)

        generator_loss = (
            generator_gan_l
            + self.lambda_1 * perceptual_l
            + self.lambda_2 * style_l
            + self.lambda_3 * content_l
            )
            # Compute the gradiants of the loss with respect to the weights of the generator
        y_hat = tf.concat([y_pred_discriminator_last_layer, y_true_discriminator_last_layer], axis=0)
        true_label = tf.concat([tf.zeros_like(y_pred_discriminator_last_layer),tf.ones_like(y_true_discriminator_last_layer)],axis=0,)
        discriminator_l = discriminator_loss(true_label, y_hat)
        # Compute the gradiants of the loss with respect to the weights of the discriminator



        # Try to do it the same way as the training step
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
        """ return {
            "style_loss": style_l,
            "content_loss": content_l,
            "perceptual_loss": perceptual_l,
            "generator_gan_loss": generator_gan_l,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_l,
        } """
    def call(self, x):
        return self.generator(x)

        
if __name__ == "__main__":
    pass
