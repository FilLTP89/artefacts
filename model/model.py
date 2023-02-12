import tensorflow as tf
from ResUNET_a_d6 import ResUNet
from Unet import Unet
from baseline import Baseline
from MedGAN import MEDGAN
from vgg19 import VGG19

class Model:
    def __init__(self, model_name, input_shape=512, learning_rate=3e-4, pretrained_vgg = True) -> None:
        super().__init__()

        self.model_name = model_name
        self.height = input_shape
        self.width = input_shape
        self.learning_rate = learning_rate
        self.pretrained_vgg = pretrained_vgg

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
            if self.pretrained_vgg:
                vgg19 = load_vgg19()
                model = MEDGAN(learning_rate = self.learning_rate, feature_extractor= vgg19)
                model.compile()
            else :
                model = MEDGAN(learning_rate = self.learning_rate)
                model.compile()
            return model
        elif self.model_name == "VGG19":
            model = VGG19(classifier_training = True)
            model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
            )
            return model

def load_vgg19():
    model = VGG19(classifier_training=False)
    model.build(input_shape = (None,512,512,1))
    model.load_weights("model/saved_models/VGG19/low_endian/VGG1902")
    for layer in model.layers:
        layer.trainable = False
    return model