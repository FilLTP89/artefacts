from data_file.processing_vgg import VGGDataset
from model.vgg19 import VGG19
import tensorflow as tf
from parsing import parse_args, default_config
import wandb
import wandb_params
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
import time
from types import SimpleNamespace


def train():
    config = SimpleNamespace(
        img_size = 512,
        batch_size = 16,
        epochs = 10,
        learning_rate = 3e-4,
        big_endian = True,
        saving_path = "model/saved_models/vgg19",
        project_name = "Training_VGG19",
    )
    wandb.init(
        project=config.project_name,
        job_type="train",
        config=config,
    )

    gpus = tf.config.list_logical_devices("GPU") if len(tf.config.list_physical_devices("GPU")) > 0 else 1
    print(f"Generating sample  with batch_size = {16 * len(gpus)}, Number of GPUs = {len(gpus)}")
    dataset = VGGDataset(
        height=config.img_size,
        width=config.img_size,
        batch_size=config.batch_size *len(gpus),
        big_endian = config.big_endian
    )
    
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Sample Generated!")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = VGG19(classifier_training = True)
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
        )
    print("Model Created!")

    callbacks = [
        WandbMetricsLogger()]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs = config.epochs,
        callbacks=callbacks,
    )
    """ y_pred = model.predict(test_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat([y for x, y in test_ds], axis=0)
    wandb.log({"test_accuracy": tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))}) """
    model.save(f"{config.saving_path}/vgg19.h5")

if __name__ == "__main__":
    train()