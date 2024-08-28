import os
import torch
import random 
import numpy as np
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module
from model.torch.Attention_MEDGAN import AttentionMEDGAN
import pytorch_lightning as pl
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)
    random.seed(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_generator", type=bool, default=True)   
    args = parser.parse_args()
    return args



def init_wandb():
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
            project=f"Metal_Artifacts_Reduction",
    )
    return wandb_logger


def init_repo(wandb_name):
    path = f"model/saved_models/torch/{wandb_name}"
    os.makedirs(path, exist_ok=True)
    return path


def load_module(*args, **kwargs):
    module = Datav2Module(*args, **kwargs)
    module.setup()
    return module

def load_model(*args, **kwargs):
    model = AttentionMEDGAN(*args, **kwargs)
    return model

def load_generator(*args, **kwargs):
    model = AttentionMEDGAN(*args, **kwargs)
    return model

def main():
    set_seed(42)
    args = init_args()
    logger.info(f"cuda is available:{torch.cuda.is_available()}, gpu available: {torch.cuda.device_count()}")
    wandb_logger = init_wandb()
    run_name = wandb_logger.experiment.name
    logger.info(run_name)
    repo_path = init_repo(run_name)
    module = load_module(
        train_bs = args.train_bs,
        test_bs = args.test_bs
    )

    if args.use_generator:
        generator = load_generator()
    model = load_model(
        learning_rate = args.lr
        )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        default_root_dir= repo_path,
        accelerator="auto", 
        devices="auto", 
        strategy="auto"
    )
    trainer.fit(model, 
                train_dataloaders = module.train_dataloader(),
                val_dataloaders = module.val_dataloader()
                )


if __name__ == "__main__":

    main()