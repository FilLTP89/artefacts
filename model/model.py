import tensorflow as tf
from ResUNET_a_d6 import ResUNet
from Unet import Unet
from baseline import Baseline
from MedGAN import MEDGAN
from vgg19 import VGG19
from loss import (
    style_loss,
    content_loss,
    perceptual_loss,
    generator_gan_loss,
    discriminator_loss,
)


class Model:
    def __init__(
        self,
        model_name,
        input_shape=512,
        learning_rate=3e-4,
        pretrained_vgg=True,
        big_endian=True,
        pretrained_MedGAN=True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.height = input_shape
        self.width = input_shape
        self.big_endian = big_endian
        self.learning_rate = learning_rate
        self.pretrained_vgg = pretrained_vgg
        self.pretrained_MedGAN = pretrained_MedGAN
        self.pretrained_MedGAN_path = (
            "model/saved_models/MedGAN/big_endian/genuine-caress-15/03"
        )
        self.pretrained_vgg_big_endian_path = (
            "model/saved_models/VGG19/big_endian/VGG1910/model.ckpt"
        )
        self.pretrained_vgg_low_endian_path = (
            "model/saved_models/VGG19/low_endian/VGG1910/model.ckpt"
        )

    def build_model(self):
        if self.model_name == "ResUnet":
            return ResUNet(
                input_shape=(self.height, self.width, 1),
                learning_rate=self.learning_rate,
                nb_class=1,
            ).build_model()
        elif self.model_name == "Unet":
            return Unet(
                input_shape=(self.height, self.width, 1),
                learning_rate=self.learning_rate,
            )
        elif self.model_name == "Baseline":
            return Baseline(
                input_shape=(self.height, self.width, 1),
            ).build_model()
        elif self.model_name == "MedGAN":
            if self.pretrained_MedGAN:
                model = load_MedGAN(self.pretrained_MedGAN_path)
            else:
                if self.pretrained_vgg:
                    print("Using pretrained VGG19")
                    vgg19 = load_vgg19(
                        big_endian=self.big_endian,
                        path=self.pretrained_vgg_big_endian_path
                        if self.big_endian
                        else self.pretrained_vgg_low_endian_path,
                    )
                    model = MEDGAN(
                        learning_rate=self.learning_rate, feature_extractor=vgg19
                    )
                else:
                    model = MEDGAN(learning_rate=self.learning_rate)
                model.compile()
                model.compute_output_shape(input_shape=(None, 512, 512, 1))
            return model
        elif self.model_name == "VGG19":
            model = VGG19(classifier_training=True)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
            )
            return model


def load_MedGAN(path=None):
    model = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "style_loss": style_loss,
            "content_loss": content_loss,
            "perceptual_loss": perceptual_loss,
            "generator_gan_loss": generator_gan_loss,
            "discriminator_loss": discriminator_loss,
        },
    )
    model.compile()
    return model


def load_vgg19(path=None):
    model = VGG19(classifier_training=False)
    model.build(input_shape=(None, 512, 512, 1))
    model.load_weights(path).expect_partial()
    for layer in model.layers:
        layer.trainable = False
    return model
