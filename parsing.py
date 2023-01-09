import argparse, os
import wandb
from pathlib import Path
import pandas as pd
from types import SimpleNamespace
import wandb_params


# defaults
default_config = SimpleNamespace(
    framework="keras",
    img_size=512,  # image size
    batch_size=8,  # Small batch size since we don't have much data
    model="resunet34d",  # resunet34d, unet
    augment=False,  # use data augmentation
    epochs=10,  # for brevity, increase for better results :)
    lr=2e-3,
    mixed_precision=False,  # use automatic mixed precision
    arch="resunet34d",
    seed=42,
    log_preds=False,
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
        "--lr", type=float, default=default_config.lr, help="learning rate"
    )
    argparser.add_argument(
        "--model",
        type=str,
        default=default_config.arch,
        help="model to train",
    )
    argparser.add_argument(
        "--augment",
        type=bool,
        default=default_config.augment,
        help="Use image augmentation",
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
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
