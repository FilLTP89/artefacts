from data_file.processing import Dataset
from model.model import Model
import tensorflow as tf
import time
import os


def train():
    seed = 42
    tf.random.set_seed(seed)
    # tf.random.get_global_generator().reset_from_seed(seed)
    print("Trying to fit overfit one batch with ResUnet")
    dataset = Dataset(
        height=512,
        width=512,
        batch_size=10,
    )
    dataset.setup()
    train_ds, valid_ds = (
        dataset.train_ds.take(32),
        dataset.valid_ds,
    )
    print("Sample Generated!")
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Creating the model ...")
        model = Model("ResUnet", 512, 3e-4).build_model()
        print("Model Created!")

    print("Start Training")
    model.fit(train_ds, validation_data=valid_ds, epochs=100, verbose=1)

    model.save("model/saved_models/overfit_one_batch/ResUnet.h5")
    print("Training Done!")


if __name__ == "__main__":
    train()
