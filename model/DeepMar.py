import tensorflow as tf
import tensorflow.keras.layers as kl


class Generator(tf.keras.Model):
    """UNET Architecture for"""

    def __init__(self, shape=(512, 512, 1)):
        super(Generator, self).__init__()
        self.shape = shape

    def down_block(self, input):
        
