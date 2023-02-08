import tensorflow as tf
import tensorflow.keras.layers as kl


class PatchGAN(tf.keras.Model):
    """
    Implementation of the PatchGAN discriminator presented in the paper
    "Image-to-Image Translation with Conditional Adversarial Networks"
    and used in the MedGan model
    """

    def __init__(self, input_shape=(512, 512, 1), patch_size=70) -> None:
        super().__init__()
        self.shape = input_shape
        self.patch_size = patch_size

        self.block_1 = tf.keras.Sequential(
            [ kl.Conv2D(64, 3, 1, padding = "same")
              kl.LeakyReLU(),
              kl.BatchNormalization()]
        )
        self.block_2 = tf.keras.Sequential(
            [ kl.Conv2D(128, 3, 1, padding = "same")
              kl.LeakyReLU(),
              kl.BatchNormalization()]
        )
    
        self.dense = kl.Dense(1, activation="sigmoid")

    def into_patch(self, input):
        bs = input.shape[0]
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Calculate the number of patches that can be extracted from the image
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        # Reshape the patches tensor to [num_patches, 16, 16, 1]
        patches = tf.reshape(
            patches, [bs, num_patches, self.patch_size, self.patch_size, 1]
        )
        return patches

    def call(self, input):
        x = self.into_patch(input)
        x1 = self.block_1(x)
        x2 = self.block2(x)
        x_ = self.dense(x2)
        return ([x, x1, x2], x_)  # return features and the result of the output layer
