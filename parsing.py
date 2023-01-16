import argparse, os
from types import SimpleNamespace


# defaults
default_config = SimpleNamespace(
    framework="keras",
    img_size=256,  # image size
    batch_size=8,  # Small batch size since we don't have much data
    model="ResUnet",  # resunet34d, unet
    augment=False,  # use data augmentation
    epochs=10,  # for brevity, increase for better results :)
    learning_rate=2e-3,
    mixed_precision=False,  # use automatic mixed precision
    log_preds=False,
    seed=42,
    wandb=False,
    save=False,
    saving_path="saved_models/",
    gpus=1,
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
        "--lr", type=float, default=default_config.learning_rate, help="learning rate"
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
        "--umin", type=int, default=137, help="Lower grayscale integer"
    )
    argparser.add_argument(
        "--umax", type=int, default=52578, help="Upper grayscale integer"
    )
    argparser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=default_config.save,
        help="Save model",
    )

    args = argparser.parse_args()
    vars(default_config).update(vars(args))
