import tensorflow as tf
import tensorflow.keras.layers as kl


"""
Implement a vgg19 model from scratch in order to control it outputs and use it as a feature extractor for the MedGAN model.
"""

# TODO: Train this model on a public dataset of medical image keyworkd : Dental CT scans
class VGG19(tf.keras.Model):
    def __init__(self, shape=(512, 512, 1)) -> None:
        super().__init__()
        self.shape = shape
        self.conv1 = kl.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.maxpool1 = kl.MaxPool2D((2, 2), strides=(2, 2))
        self.conv2 = kl.Conv2D(128, (3, 3), activation="relu", padding="same")
        self.maxpool2 = kl.MaxPool2D((2, 2), strides=(2, 2))
        self.conv3 = kl.Conv2D(256, (3, 3), activation="relu", padding="same")
        self.conv4 = kl.Conv2D(256, (3, 3), activation="relu", padding="same")
        self.maxpool3 = kl.MaxPool2D((2, 2), strides=(2, 2))
        self.conv5 = kl.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv6 = kl.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.maxpool4 = kl.MaxPool2D((2, 2), strides=(2, 2))
        self.conv7 = kl.Conv2D(512, (3, 3), activation="relu", padding="same")
        self.conv8 = kl.Conv2D(512, (3, 3), activation="relu", padding="same")

        self.maxpool5 = kl.MaxPool2D((2, 2), strides=(2, 2))
        self.flatten = kl.Flatten()
        self.dense1 = kl.Dense(4096, activation="relu")
        self.dense2 = kl.Dense(4096, activation="relu")
        self.output_layer = kl.Dense(1000, activation="softmax")

    def call(self, input):
        x = self.conv1(input)
        x2 = self.maxpool1(x)
        x2 = self.conv2(x2)
        x3 = self.maxpool2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x3)
        x5 = self.maxpool3(x4)
        x5 = self.conv5(x5)
        x6 = self.conv6(x5)
        x7 = self.maxpool4(x6)
        x7 = self.conv7(x7)
        x8 = self.conv8(x7)

        x9 = self.maxpool5(x8)
        x9 = self.flatten(x9)
        x9 = self.dense1(x9)
        x9 = self.dense2(x9)
        output = self.output_layer(x9)

        return ([input, x, x2, x3, x4, x5, x6, x7, x8], output)
        # return the list of outut of each convolutional layer
