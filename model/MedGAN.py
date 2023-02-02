import tensorflow as tf
import tensorflow.keras.layers as kl


class U_block(tf.keras.Model):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape

    def down_conv_block(self, input, filters, kernel_size=4, strides=2):
        x = kl.Conv2D(filters, kernel_size, strides)(input)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU()(x)
        return x

    def up_conv_block(
        self, input, skip_input, filters, kernel_size=4, strides=2, padding="same"
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
        u1, u2, u3, u4, u5, u6, u7, u8 = encoder_list

        for i, filter in enumerate(filter):
            if i == 0:
                y = self.up_conv_block(encoder_list[i - 1], encoder_list[i - 2], filter)

        return d8

    def build_model(self):
        input = kl.Input(shape=self.shape)
        encoder_list = self.encoder_block(input)
        output = self.decoder_block(encoder_list)
        model = tf.keras.Model(input, output)
        return model


class PatchGAn(tf.keras.Model):

    pass


class MEDGAN(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        learning_rate=3e-4,
        generator=None,
        discriminator=None,
        loss=None,
    ):
        super().__init__()
        self.shape = input_shape
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics_list = [tf.keras.metrics.RootMeanSquaredError()]

        generator = generator or self.Generator()
        discriminator = discriminator or self.Discriminator()

    def Generator(self):
        """ """

    def Discriminator(self):
        return


if __name__ == "__main__":
    U_block = U_block((256, 256, 1))
    x = tf.random.normal((1, 256, 256, 1))
    model = U_block.build_model()
    y = model(x)

    print(y.shape)
