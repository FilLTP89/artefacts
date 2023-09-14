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
    MSE_loss
)



class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):

        self.W_gate = tf.keras.Sequential(
            [ kl.Conv2D(channels, 1, 1, padding="same"), 
             kl.BatchNormalization()]
        )
        self.W_x = tf.keras.Sequential(
            [ kl.Conv2D(channels, 1, 1, padding="same"),
                kl.BatchNormalization()]
        )
        self.phi = tf.keras.Sequential(
            [ kl.Conv2D(channels, 1, 1, padding="same"),
                kl.BatchNormalization()]
        )

        self.attention = kl.Attention()

    def call(self, x, g ):
        g = self.W_gate(g)
        x = self.W_x(x)
        phi = g + x
        phi = kl.Activation("relu")(phi)
        phi = self.phi(phi)
        phi = kl.Activation("sigmoid")(phi)
        return self.attenion([x, phi])

       


class U_block(tf.keras.Model):

    def __init__(self, shape=(512, 512, 1)) -> None:
        super().__init__()
        self.shape = shape
        self.filters = [32,64,128,256,512,1024]
        self.attention_layers = [SelfAttention(filter) for filter in self.filters]

    def original_conv(self, input, filters = 32, kernel_size=1, strides=1, padding="same"):
        x = kl.Conv2D(
            filters,
            kernel_size,
            strides,
            padding,
        )(input)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU()(x)
        return x

    def last_conv(self, input, filters = 1, kernel_size=1, strides=1, padding="same"):
        x = kl.Conv2D(
            filters,
            kernel_size,
            strides,
            padding,
        )(input)
        return x
    
    def skip_connection(self, input, skip_input):
        """
        Modify this with the attention layer 
        """
        x = kl.Concatenate()([input, skip_input])
        return x
    

    def down_conv_block(self, input, filters, kernel_size=4, strides=2, padding="same"):
        x = kl.Conv2D(
            filters,
            kernel_size,
            strides,
            padding,
        )(input)
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
        x = self.skip_connection(x, skip_input)
        return x

    def bottele_neck_block(self, input, filters=2048):
        x = kl.Conv2D(filters, kernel_size = 4, strides = 2, padding="same")(input)
        x = kl.BatchNormalization()(x)
        x = kl.ReLU()(x)
        return x
    
    
    def encoder_block(self, input):
        x = input
        encoder_list = [x]
        for filter in self.filters:
            x = self.down_conv_block(x, filter)
            encoder_list.append(x)
        return encoder_list



    def decoder_block(self, encoder_list):
        last_layer = encoder_list[-1]
        encoder_list = encoder_list[::-1]
        y = self.bottele_neck_block(last_layer)
        for i, (filter,encoded) in enumerate(zip(self.filters[::-1], encoder_list)):
            y = self.up_conv_block(y, encoded, filter)
        y = kl.Conv2DTranspose(1, 4, 2, padding="same")(y)
        y = kl.Concatenate()([y, encoder_list[-1]])
        return y

    def build_model(self):
        input = kl.Input(shape=self.shape)
        x = self.original_conv(input)
        encoder_list = self.encoder_block(x)
        decoder_output = self.decoder_block(encoder_list)
        output = self.last_conv(decoder_output)
        model = tf.keras.Model(inputs=input, outputs=output)
        return model


class ConsNet(tf.keras.Model):
    def __init__(self, n_block=6, input_shape=(512, 512, 1)) -> None:
        super().__init__()
        self.shape = input_shape
        self.Ublock = [U_block(self.shape).build_model() for _ in range(n_block)]

    def call(self, inputs, training=False):
        x = inputs
        for block in self.Ublock:
            x = block(x)
        y = kl.Activation("sigmoid")(x)
        return y


class AttentionMEDGAN(tf.keras.Model):
    """
    Implementation of the MedGAN model presented in the paper
    """

    def __init__(
        self,
        input_shape=(512, 512, 1),
        generator=None,
        discriminator=None,
        feature_extractor=None,
        learning_rate=3e-5,
        N_g=3,
        vgg_whole_arc=False,
    ):
        super().__init__()

        self.shape = input_shape

        self.N_g = N_g  # number of training iterations for generator
        # TO DO : Do training as 1-1 and change the learning rate of the generator to be higher ?

        self.lambda_1 = 20
        self.lambda_2 = 1e-4
        self.lambda_3 = 1

        # self.learning_rate = learning_rate
        self.learning_rate = learning_rate

        self.g_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate,
            ema_momentum=0.5,
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate,
            ema_momentum=0.5,
        )

        self.generator = generator or ConsNet(3, self.shape)
        self.discriminator = discriminator or PatchGAN(self.shape)
        self.feature_extractor = feature_extractor or VGG19(
            self.shape, load_whole_architecture=vgg_whole_arc
        )

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
        self.mse_loss_tracker = tf.keras.metrics.Mean(name="MSE_loss")

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.perceptual_loss_tracker,
            self.generator_gan_loss_tracker,
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.mse_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data

        for _ in range(self.N_g):  # N_g = 3
            with tf.GradientTape() as gen_tape:
                y_pred = self.generator(x)
                (
                    y_pred_discriminator_features,
                    y_pred_discriminator_last_layer,
                ) = self.discriminator(y_pred)
                # get the features of the discriminator and the output of the last layer for the output of the generator

                y_pred_feature = self.feature_extractor(y_pred)
                # get the features of the feature extractor for the true samples
                y_true_feature = self.feature_extractor(y)
                (y_true_discriminator_features, _) = self.discriminator(y)
                # get the features of the discriminator and the output of the last layer for the true samples
                # gan loss
                generator_gan_l = generator_gan_loss(y_pred_discriminator_last_layer)
                # perceptual loss
                perceptual_l = perceptual_loss(
                    y_true_discriminator_features, y_pred_discriminator_features
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
            gen_grads = gen_tape.gradient(
                generator_loss, self.generator.trainable_weights
            )
            # Update the weights of the generator
            self.g_optimizer.apply_gradients(
                zip(gen_grads, self.generator.trainable_weights)
            )

        with tf.GradientTape() as disc_tape:
            _, y_pred_discriminator = self.discriminator(y_pred)
            _, y_true_discriminator = self.discriminator(y)
            # Discriminator loss
            discriminator_l = discriminator_loss(
                y_true_discriminator, y_pred_discriminator
            )

            # Compute the gradiants of the loss with respect to the weights of the discriminator
        disc_grads = disc_tape.gradient(
            discriminator_l, self.discriminator.trainable_weights
        )

        # Update the weights of the discriminator
        self.d_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_weights)
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
        ) = self.discriminator(y_pred)
        # get the features of the discriminator and the output of the last layer for the output of the generator
        y_pred_feature = self.feature_extractor(
            y_pred
        )  # get the features of the feature extractor for the true samples
        y_true_feature = self.feature_extractor(y)
        (
            y_true_discriminator_features,
            y_true_discriminator_last_layer,
        ) = self.discriminator(y)
        # get the features of the discriminator and the output of the last layer for the true samples

        # gan loss
        generator_gan_l = generator_gan_loss(y_pred_discriminator_last_layer)
        # perceptual loss
        perceptual_l = perceptual_loss(
            y_true_discriminator_features, y_pred_discriminator_features
        )
        # style loss
        style_l = style_loss(y_true_feature, y_pred_feature)
        # content loss
        content_l = content_loss(y_true_feature, y_pred_feature)

        generator_loss = (
            generator_gan_l
            + self.lambda_1 * perceptual_l
            + self.lambda_2 * style_l
            + self.lambda_3 * content_l
        )
        # Compute the gradiants of the loss with respect to the weights of the generator
        discriminator_l = discriminator_loss(
            y_true_discriminator_last_layer, y_pred_discriminator_last_layer
        )
        # Compute the gradiants of the loss with respect to the weights of the discriminator
        mse_l = MSE_loss(y, y_pred) 
        # Try to do it the same way as the training step
        self.style_loss_tracker.update_state(style_l)
        self.content_loss_tracker.update_state(content_l)
        self.perceptual_loss_tracker.update_state(perceptual_l)
        self.generator_gan_loss_tracker.update_state(generator_gan_l)
        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_l)
        self.mse_loss_tracker.update_state(mse_l)

        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "generator_gan_loss": self.generator_gan_loss_tracker.result(),
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
            "MSE_loss": self.mse_loss_tracker.result(),
        }

    def call(self, x):
        y = self.generator(x)
        y_pred_discriminator = self.discriminator(y)
        y_pred_feature = self.feature_extractor(y)
        return y


if __name__ == "__main__":
    model = AttentionMEDGAN()
    #model = U_block().build_model()
    y = model(tf.random.normal((2, 512, 512, 1)))
    #print(y.shape)
    model.summary()
