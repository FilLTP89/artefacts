import wandb
import yaml
from pathlib import Path
import tensorflow as tf
from data_file.processing import Dataset
from model.model import Model
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
import argparse


def train():
    wandb.init()
    config = wandb.config
    tf.random.set_seed(config.seed)
    print(
        f"Generating sample  with batch_size = {config.batch_size * config.gpus}, big_endian = {config.big_endian}"
    )
    dataset = Dataset(
        height=config.img_size,
        width=config.img_size,
        batch_size=config.batch_size * config.gpus,
        big_endian=config.big_endian,
    )
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Sample Generated!")
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Creating the model ...")
        model = Model(config.model, config.img_size, config.learning_rate).build_model()
        print("Model Created!")
    print(f"save = {config.save}")
    if config.save:
        callbacks = callbacks = [
            WandbMetricsLogger(),
            WandbModelCheckpoint(
                config.saving_path + config.name + ".h5",
            ),
        ]
    else:
        callbacks = [WandbMetricsLogger()]
    print("Start Training")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.epochs,
        verbose=1,
        callbacks=callbacks,
    )
    print("Training Done!")


def sweep(sweep_config=None):
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
    wandb.agent(sweep_id, function=train, count=50)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a model")
    argparser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the model",
    )
    argparser.add_argument(
        "--big_endian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the data is big endian or not",
    )
    yaml_dict = yaml.safe_load(Path("sweep.yaml").read_text())
    yaml_dict["parameters"]["big_endian"]["value"] = argparser.parse_args().big_endian
    yaml_dict["parameters"]["save"]["value"] = argparser.parse_args().save
    if argparser.parse_args().big_endian:
        yaml_dict["project"] = "artefact-detection-big_endian"
    else:
        yaml_dict["project"] = "artefact-detection-low_endian"
    sweep(yaml_dict)
