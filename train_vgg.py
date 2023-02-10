from data_file.processing_vgg import VGGDataset
from model.vgg19 import VGG19
import tensorflow as tf
from parsing import parse_args, default_config
import wandb
import wandb_params
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
import time



def train():
    gpus = 
    dataset = VGGDataset(
        height=512,
        width=512,
        batch_size=config.batch_size * len(gpus),
        big_endian = True
    )
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    model = VGG19(classifier_training = True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs = 10
    )
    model.save("model/saved_models/vgg19/vgg19.h5")

if __name__ == "__main__"
    train()