# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Dropout,
    Lambda,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    ReLU,
    concatenate,
)


def Unet(input_shape, learning_rate, **kwargs):

    # Build U-Net model
    # inputs = Input((imageHeight, imageWidth, imageChannels))
    inputs = Input(input_shape)
    s = Lambda(lambda x: x / 255)(inputs)

    # Block 1
    c1 = Conv2D(
        2, (2, 2), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)

    # Block 2
    c2 = Conv2D(4, (2, 2), kernel_initializer="he_normal", padding="same")(c1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU()(c2)

    # Block 3
    c3 = Conv2D(8, (2, 2), kernel_initializer="he_normal", padding="same")(c2)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU()(c3)

    # Block 4
    c4 = Conv2D(16, (2, 2), kernel_initializer="he_normal", padding="same")(c3)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU()(c4)

    # Block 5
    c5 = Conv2D(16, (2, 2), kernel_initializer="he_normal", padding="same")(c4)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU()(c5)

    # Block 6
    c6 = Conv2D(16, (2, 2), kernel_initializer="he_normal", padding="same")(c5)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU()(c6)

    # Block 7
    u7 = Conv2DTranspose(32, (2, 2), padding="same")(c6)
    u7 = concatenate([u7, c5])
    c7 = BatchNormalization()(u7)
    c7 = ReLU()(c7)

    # Block 8
    u8 = Conv2DTranspose(32, (2, 2), padding="same")(c7)
    u8 = concatenate([u8, c4])
    c8 = BatchNormalization()(u8)
    c8 = ReLU()(c8)

    # Block 9
    u9 = Conv2DTranspose(16, (2, 2), padding="same")(c8)
    u9 = concatenate([u9, c3])
    c9 = BatchNormalization()(u9)
    c9 = ReLU()(c9)

    # Block 10
    u10 = Conv2DTranspose(8, (2, 2), padding="same")(c9)
    u10 = concatenate([u10, c2])
    c10 = BatchNormalization()(u10)
    c10 = ReLU()(c10)

    # Block 11
    u11 = Conv2DTranspose(4, (2, 2), padding="same")(c10)
    u11 = concatenate([u11, c1])
    c11 = Dropout(0.1)(u11)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


if __name__ == "__main__":
    model = Unet(400, 400, 1)
    model.summary()
