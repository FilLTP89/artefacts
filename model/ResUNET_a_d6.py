import tensorflow as tf
import tensorflow.keras.layers as kl


class ResUNet(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        nb_class,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.shape = input_shape  # Input shape has to be power of 2, 256x256 or 512x512
        self.nb_class = nb_class
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metric = [
            tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
        ]

    def ResBlock_a(
        self,
        x,
        filters: int,
        dilation_list: list,
        kernel_size=3,
        padding: str = "same",
        strides: int = 1,
    ):

        """
        Resblock-a described in the paper, pass through a sequence of convolution layers,
        the output of the parrallel sequence is then stacked and sum.

        Args
        -----
        x (Tensor): input of the block
        filters (int) : the number of filters for the convolutional layers.
        dilatation_list (list) : contain the different dilatation rate for the parrallele sequence.
        kernel_size (int or tuple) : the size of the kernel of the convolutional layers.
        padding (str) : the padding for the convolutional layers.
        stride (int) : the stride for the convolution layers.

        Returns
        -----
        output (Tensor) : output of the block
        """

        output = [x]
        for d in dilation_list:
            x_ = kl.BatchNormalization(axis=-1)(x)
            x_ = kl.Activation("relu")(x_)
            x_ = kl.Conv2D(
                filters,
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                dilation_rate=d,
            )(x_)
            x_ = kl.BatchNormalization(axis=-1)(x_)
            x_ = kl.Activation("relu")(x_)
            x_ = kl.Conv2D(
                filters,
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                dilation_rate=d,
            )(x_)
            output.append(x_)
        output = tf.stack(output, axis=0)
        return tf.math.reduce_sum(output, axis=0)

    def Combine(
        self, x1, x2, filters, kernel_size=1, strides=1, padding="same", upsample=True
    ):
        """
        Used in the up_part of the model, the tensor x1 output of previous layers is upsampled and concatened
        to a tensor of the down_part. The concatenated tensor is then passed to a convolutional and batch norm
        layer.

        Args
        ----
        x1 (Tensor) : output of previous layer
        x2 (Tensor) : one of the ouput of the down part
        filters (int) : the number of filters

        Returns
        -----
        x (Tensor) :  output of the sequence described
        """
        if upsample:
            x1 = kl.UpSampling2D(size=2)(x1)
        x1 = kl.Activation("relu")(x1)
        x = tf.concat([x1, x2], axis=-1)
        x = kl.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )(x)
        x = kl.BatchNormalization(axis=-1)(x)
        return x

    def down_part(self, x):
        """
        Down (or encoder) part of the model, correspond to layers 1 to 12.

        Args
        -----

        Returns
        -----
        """
        x = kl.Conv2D(32, 1, padding="same")(x)  # Layer 1

        u1 = self.ResBlock_a(x, 32, [1, 3, 15, 31])  # Layer 2

        u2 = kl.Conv2D(64, 1, padding="same", strides=2)(u1)  # Layer 3
        u2 = self.ResBlock_a(u2, 64, [1, 3, 15, 31])  # Layer 4

        u3 = kl.Conv2D(128, 1, padding="same", strides=2)(u2)  # Layer 5
        u3 = self.ResBlock_a(u3, 128, [1, 3, 15])  # Layer 6

        u4 = kl.Conv2D(256, 1, padding="same", strides=2)(u3)  # Layer 7
        u4 = self.ResBlock_a(u4, 256, [1, 3, 15])  # Layer 8

        u5 = kl.Conv2D(512, 1, padding="same", strides=2)(u4)  # Layer 9
        u5 = self.ResBlock_a(u5, 512, [1])  # Layer 10

        u6 = kl.Conv2D(1024, 1, padding="same", strides=2)(u5)  # Layer 11
        u6 = self.ResBlock_a(u6, 1024, [1])  # Layer 12

        return x, u1, u2, u3, u4, u5, u6

    def PSP_Pooling(self, x, filter):  # Layer 13

        """
        PSP Pooling (Layer 13) part described in the paper.
        Description : "  In the PSPPooling operator,
        the initial input is split in channel (feature) space in
        4 equal partitions. Then we perform max pooling operation in succes-
        sive splits of the input layer, in 1, 4, 16 and 64 partitions"

        Args
        -----
        x (Tensor) : Input of the Layers

        """

        output = [x]
        split = tf.split(
            x, 4, axis=-1
        )  # the initial input is split in channel (feature) space in 4 equal partitions
        for elem, rate in zip(split, [1, 2, 4, 8]):
            elem = kl.MaxPool2D(pool_size=rate, strides=rate)(elem)
            elem = kl.UpSampling2D(size=rate)(elem)
            elem = kl.Conv2D(filter // 4, 1, padding="same")(elem)
            output.append(elem)
        concat = tf.concat([elem], axis=-1)
        concat = kl.Conv2D(filter, 1, padding="same")(concat)
        return concat

    def up_part(self, x_firstconv, u1, u2, u3, u4, u5, x_bottleneck):

        x = self.Combine(x_bottleneck, u5, 512)  # Layer 14 & 15
        x = self.ResBlock_a(x, 512, [1])  # Layer 16

        x = self.Combine(x, u4, 256)  # Layer 17 & 18
        x = self.ResBlock_a(x, 256, [1, 3, 15])  # Layer 19

        x = self.Combine(x, u3, 128)  # Layer 20 & 21
        x = self.ResBlock_a(x, 128, [1, 3, 15])  # Layer 22

        x = self.Combine(x, u2, 64)  # Layer 23 & 24
        x = self.ResBlock_a(x, 64, [1, 3, 15, 31])  # Layer 25

        x = self.Combine(x, u1, 32)  # Layer 26 & 27
        x = self.ResBlock_a(x, 32, [1, 3, 15, 31])  # Layer 28

        x = self.Combine(x, x_firstconv, 32, upsample=False)  # Layer 29

        x = self.PSP_Pooling(x, 32)  # Layer 30

        if self.nb_class == 1:
            x = kl.Conv2D(1, 1, activation="sigmoid")(x)  # Layer 31
        else:
            x = kl.Conv2D(self.nb_class, 1, activation="softmax")(x)  # Layer 31

        return x

    def build_model(self):
        input = kl.Input(shape=self.shape)
        x_firstconv, u1, u2, u3, u4, u5, u6 = self.down_part(input)
        x_bottleneck = self.PSP_Pooling(u6, 1024)
        output = self.up_part(x_firstconv, u1, u2, u3, u4, u5, x_bottleneck)
        model = tf.keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metric
        )  # compile the model

        return model


if __name__ == "__main__":
    model = ResUNet(
        (256, 256, 1), 1
    ).build_model()  # width and height have to multiple of power of 2
    model.summary()
    # model.save("ResUNet.h5")  # save the model
