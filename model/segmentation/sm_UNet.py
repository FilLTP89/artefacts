from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import tensorflow as tf




class sm_UNet(tf.keras.Model):
    def __init__(self, 
                 input_shape = (512,512,1),
                  nb_class = 1,
                  learning_rate = 3e-4, ):
        super().__init__()
        self.shape = input_shape  # Input shape has to be power of 2, 256x256 or 512x512
        self.nb_class = nb_class
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_mode(self):
        model = Unet(backbone_name='resnet34', encoder_weights='imagenet')
        model.compile(optimizer=self.optimizer, loss=bce_jaccard_loss, metrics=[iou_score])
        return model
    

if __name__ == "__main__":
    model = sm_UNet()
    model.build_mode()
    x = tf.random.uniform((1, 512, 512, 1))   
    y = model(x)
    y_true = tf.random.uniform((1, 512, 512, 1))
    print(model.loss(y_true, y))   
