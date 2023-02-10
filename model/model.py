import tensorflow as tf
from ResUNET_a_d6 import ResUNet
from Unet import Unet
from baseline import Baseline
from MedGAN import MEDGAN

class Model:
    def __init__(self, model_name, input_shape=512, learning_rate=3e-4, pretrained_vgg = False) -> None:
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
                vgg = tf.keras.models.load_model("saved_models/vgg19/vgg19.h5")
                # Freeze the layers
                for layer in vgg.layers:
                    layer.trainable = False
                model = MEDGAN(learning_rate = self.learning_rate, feature_extractor= vgg)
                model.compile()
            else :
                model = MEDGAN(learning_rate = self.learning_rate, feature_extractor= vgg)
                model.compile()
            return model
