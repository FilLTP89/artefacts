import os
import tensorflow as tf
from data_file.processing_segmentation import SegmentationDataset
from model.segmentation.ResUNET_a_d6 import ResUNet
import matplotlib.pyplot as plt



def load_model(
    model_path = "/gpfs/workdir/candemilam/artefacts/model/saved_models/ResUnet/low_endian/revived-firefly-13/16"
    ): 
    model_path = "/gpfs/workdir/candemilam/artefacts/model/saved_models/ResUnet/low_endian/confused-glitter-14/200" #New try
    try:
        model = tf.keras.models.load_model(model_path)
        model.compile()
    except:
        model_path += "/model.ckpt"
        model = ResUNet().build_model()
        #model.build(input_shape=(None, 512, 512, 1))
        model.load_weights(model_path).expect_partial()
    return model


def generate(batch_size=32,):
    dataset = SegmentationDataset(batch_size=batch_size,)
    dataset.setup()
    model = load_model()
    d = 0
    while os.path.exists(f"generated_images/segmentation/experiment_{d}"):
        d += 1
    os.makedirs(f"generated_images/segmentation/experiment_{d}")
    data = dataset.train_ds
    for _, (x, y) in enumerate(data):
        preds = tf.round(model.predict(x))
        for i in range(preds.shape[0]):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))  
            ax1.imshow(x[i], cmap="gray")
            ax2.imshow(y[i], cmap="gray")
            ax3.imshow(preds[i], cmap="gray")
            ax1.set_title("Input")
            ax2.set_title("Ground Truth")
            ax3.set_title("Prediction")
            plt.savefig(f"generated_images/segmentation/experiment_{d}/{i}.png")
            plt.close()
        break


if __name__ == "__main__":
    generate(batch_size=32)
