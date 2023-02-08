import tensorflow as tf
from ResUNET_a_d6 import ResUNet
from Unet import Unet
from baseline import Baseline
from MedGAN import MEDGAN

class Model:
    def __init__(self, model_name, input_shape=512, learning_rate=3e-4) -> None:
        super().__init__()

        self.model_name = model_name
        self.height = input_shape
        self.width = input_shape
        self.learning_rate = learning_rate

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
            model = MEDGAN()
            model.compile()
            return model
