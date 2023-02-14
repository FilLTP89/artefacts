import argparse, os
from types import SimpleNamespace


# defaults
default_config = SimpleNamespace(
    img_size=512,  # image size has to be a power of two for resunet
    batch_size=8,  # Small batch size since we don't have much data
    model="ResUnet",  # resunet34d, unet
    augment=False,  # use data augmentation
    epochs=10,
    learning_rate=2e-6,
    log_preds=False,
    seed=42,
    wandb=False,
    save=False,
    saving_path="model/saved_models/",
    gpus=1,
    mixed_precision=False,  # use automatic mixed precision -> fasten training
    run_name="training_run",
    big_endian=False,
    one_batch_training=False
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument(
        "--img_size", type=int, default=default_config.img_size, help="image size"
    )
    argparser.add_argument(
        "--batch_size", type=int, default=default_config.batch_size, help="batch size"
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=default_config.epochs,
        help="number of training epochs",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=default_config.learning_rate,
        help="learning rate",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default=default_config.model,
        help="model to train",
    )
    argparser.add_argument(
        "--augment",
        type=bool,
        default=default_config.augment,
        help="Use image augmentation",
    )
    argparser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=default_config.wandb,
        help="Use wandb",
    )
    argparser.add_argument(
        "--big_endian",
        action=argparse.BooleanOptionalAction,
        default=default_config.big_endian,
        help="Use big endian",
    )
    argparser.add_argument(
        "--seed", type=int, default=default_config.seed, help="random seed"
    )
    argparser.add_argument(
        "--log_preds",
        type=bool,
        default=default_config.log_preds,
        help="log model predictions",
    )
    argparser.add_argument(
        "--gpus",
        type=int,
        default=default_config.gpus,
        help="number of gpus to use",
    )
    argparser.add_argument(
        "--mixed_precision",
        type=bool,
        default=default_config.mixed_precision,
        help="use fp16",
    )
    argparser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=default_config.save,
        help="Save model",
    )
    argparser.add_argument(
        "--run_name",
        default=default_config.run_name,
        help="Name of the run",
    )
    argparser.add_argument(
        "--saving_path",
        default=default_config.saving_path,
        help="Path to save model",
    )
    argparser.add_argument(
        "--one_batch_training",
        action=argparse.BooleanOptionalAction,
        default=default_config.one_batch_training,
        help="Train on one batch",
    )

    args = argparser.parse_args()
    vars(default_config).update(vars(args))
