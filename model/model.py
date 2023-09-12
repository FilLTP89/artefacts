import tensorflow as tf
from ResUNET_a_d6 import ResUNet
from Unet import Unet
from baseline import Baseline
from MedGAN import MEDGAN
from vgg19 import VGG19
from DeepMar import DeepMar
from loss import (
    style_loss,
    content_loss,
    perceptual_loss,
    generator_gan_loss,
    discriminator_loss,
)
from segmentation.ResUNET_a_d6 import ResUNet as smResunet

class Model:
    def __init__(
        self,
        model_name,
        input_shape=512,
        learning_rate=3e-4,
        pretrained_vgg=True,
        big_endian=True,
        dicom=True,
        pretrained_MedGAN=True,
        segmentation=False,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.height = input_shape
        self.width = input_shape
        self.big_endian = big_endian
        self.dicom = dicom
        self.learning_rate = learning_rate
        self.pretrained_vgg = pretrained_vgg
        self.pretrained_MedGAN = pretrained_MedGAN
        #self.pretrained_MedGAN_path = ("model/saved_models/MedGAN/big_endian/genuine-caress-15/03")
        self.pretrained_MedGAN = "model/saved_models/MedGAN/big_endian/vibrant-dawn-3/40"
        self.pretrained_vgg_big_endian_path = (
            "model/saved_models/VGG19/big_endian/VGG1910/model.ckpt"
        )
        self.pretrained_vgg_low_endian_path = (
            "model/saved_models/VGG19/low_endian/VGG1910/model.ckpt"
        )
        self.pretrained_vgg_dicom_path = "model/saved_models/VGG19/dicom/grateful-capybara-8/20/model.ckpt"  # acc : dicom normalize between -1 and 1

        self.segmentation = segmentation


    def build_model(self):
        if self.model_name == "ResUnet":
            model = ResUNet(
                input_shape=(self.height, self.width, 1),
                learning_rate=self.learning_rate,
                nb_class=1,
            ).build_model()
        elif self.model_name == "Unet":
            model = Unet(
                input_shape=(self.height, self.width, 1),
                learning_rate=self.learning_rate,
            )
        elif self.model_name == "Baseline":
            model = Baseline(
                input_shape=(self.height, self.width, 1),
            ).build_model()
        elif self.model_name == "MedGAN":
            if self.pretrained_MedGAN:
                print("Using Pretrained MedGAN")
                model = load_MedGAN_from_checkpoint(self.pretrained_MedGAN)
            else:
                if self.pretrained_vgg:
                    print("Using pretrained VGG19")
                    if self.dicom:
                        vgg19 = load_vgg19(path=self.pretrained_vgg_dicom_path)
                    elif self.big_endian:
                        vgg19 = load_vgg19(path=self.pretrained_vgg_big_endian_path)
                    else:
                        vgg19 = load_vgg19(path=self.pretrained_vgg_low_endian_path)
                    model = MEDGAN(
                        learning_rate=self.learning_rate, feature_extractor=vgg19
                    )
                else:
                    model = MEDGAN(learning_rate=self.learning_rate)
                model.compile()
                model.compute_output_shape(input_shape=(None, 512, 512, 1))
        elif self.model_name == "VGG19":
            model = VGG19(classifier_training=True)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
            )
        elif self.model_name == "DeepMAR":
            model = DeepMar(
                learning_rate=self.learning_rate,
            )
            model.compile()
            model.compute_output_shape(input_shape=(None, 512, 512, 1))


        elif self.model_name == "smResunet":
            model = smResunet(
                input_shape=(self.height, self.width, 1),
                learning_rate=self.learning_rate,
                nb_class=1,
            ).build_model()
        return model


def load_MedGAN(path=None):
    custom_objects = {
        "style_loss": style_loss,
        "content_loss": content_loss,
        "perceptual_loss": perceptual_loss,
        "generator_gan_loss": generator_gan_loss,
        "discriminator_loss": discriminator_loss,
    }

    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    model.compile(
        loss=[
            style_loss,
            content_loss,
            perceptual_loss,
            generator_gan_loss,
            discriminator_loss,
        ]
    )
    return model


def load_MedGAN_from_checkpoint(path=None):
    try : 
        path += "/model.ckpt"
        model = MEDGAN(learning_rate=5e-5)
        model.build(input_shape=(None, 512, 512, 1))
        model.load_weights(path).expect_partial()
        print("MedGAN loaded")
    except :
        print("Error while loading the model")
    return model

def load_vgg19(path=None):
    model = VGG19( load_whole_architecture = False, classifier_training= False)
    model.build(input_shape=(None, 512, 512, 1))
    model.load_weights(path).expect_partial()
    #model = tf.keras.models.load_model(path)
    print("VGG19 loaded")
    for layer in model.layers:
        layer.trainable = False
    return model


if __name__ == "__main__":
    model = load_vgg19(
        path="model/saved_models/VGG19/dicom/logical-haze-1/09/model.ckpt"
    )
    model.summary()
