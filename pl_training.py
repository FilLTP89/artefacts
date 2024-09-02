import os
import torch
import random 
import numpy as np
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module, Datav2Dataset, ClassificationDataset
from model.torch.Attention_MEDGAN import AttentionMEDGAN, VGG19
import pytorch_lightning as pl
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



SAVE_WEIGHTS_ONLY = True

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
    parser.add_argument("--use_generator", action=argparse.BooleanOptionalAction,type=bool, default=True)   
    parser.add_argument("--one_batch",action=argparse.BooleanOptionalAction, type=bool, default=False)    
    parser.add_argument("--mix_precision", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--ruche", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--task", type=str, default="GAN")
    args = parser.parse_args()
    return args



def init_wandb():
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
            project=f"Metal_Artifacts_Reduction",
    )
    return wandb_logger


def init_repo(wandb_name, ruche = False):
    print(ruche)
    if not ruche:
        path = f"model/saved_model/{wandb_name}"
        os.makedirs(path, exist_ok=True)
    else:
        path = f"/gpfs/users/gabrielihu/saved_model/{wandb_name}"
        os.makedirs(path, exist_ok=True)
    return path


def load_module(
        task = "GAN",
        *args, **kwargs):
    if task == "GAN":
        module = Datav2Module(dataset_type= Datav2Dataset,*args, **kwargs)
    else:
        module = Datav2Module(dataset_type= ClassificationDataset,*args, **kwargs)
    module.setup()
    return module

def load_model(task ="GAN",
               *args, **kwargs):
    if task == "GAN":
        model = AttentionMEDGAN(*args, **kwargs)
    return model

def load_generator(*args, **kwargs):
    model = AttentionMEDGAN(*args, **kwargs)
    return model

def main():
    set_seed(42)
    device_count = torch.cuda.device_count()
    args = init_args()
    wandb_logger = init_wandb()
    run_name = wandb_logger.experiment.name
    repo_path = init_repo(run_name, args.ruche)
    module = load_module(
        train_bs = args.train_bs,
        test_bs = args.test_bs
    )

    
    rank_zero_info("\n \n \n ")
    rank_zero_info(f"Cuda is available:{torch.cuda.is_available()}, gpu available: {device_count}")
    rank_zero_info(f"Run name {run_name}")
    rank_zero_info(f"Run path {repo_path}")
    rank_zero_info(f"Use mix precision : {args.mix_precision}")
    rank_zero_info(f"Use one batch : {args.one_batch}")
    rank_zero_info(f"Save weight only : {SAVE_WEIGHTS_ONLY}")
    rank_zero_info("\n \n \n ")

    if args.use_generator:
        generator = load_generator()
    model = load_model(
        learning_rate = args.lr
        )
    callbacks = [
            ModelCheckpoint(
        dirpath = repo_path,
        filename = "best_model-{epoch:02d}-{test_mse_loss:.2f}",
        save_top_k =1,
        verbose = True,   
        monitor = "test_mse_loss",
        mode = "min",
        save_weights_only=SAVE_WEIGHTS_ONLY
        ),
        LearningRateMonitor(logging_interval='step')]
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator="gpu", 
        devices=device_count, 
        strategy="ddp_find_unused_parameters_true",
        overfit_batches= 1 if args.one_batch else 0,
        num_nodes=1,
        callbacks=callbacks,
        precision=16 if args.mix_precision else 32,
        log_every_n_steps=1,
    )
    trainer.fit(model, 
                train_dataloaders = module.train_dataloader(),
                val_dataloaders = module.val_dataloader()
                )


if __name__ == "__main__":
    main()