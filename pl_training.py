import os
import torch
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module
from model.torch.Attention_MEDGAN import AttentionMEDGAN
import pytorch_lightning as pl
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRAIN_BS = 16
TEST_BS = 16
MAX_EPOCHS = 100

def init_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()
    return args



def init_wandb():
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
            project=f"Metal_Artifacts_Reduction",
    )
    return wandb_logger


def init_repo(wandb_name):
    path = f"model/saved_model/New_data/{wandb_name}"
    os.makedirs(path, exist_ok=True)
    return path


def load_module():
    module = Datav2Module(train_bs=TRAIN_BS,
                          test_bs=TEST_BS)
    module.setup()
    return module

def load_model():
    model = AttentionMEDGAN()
    return model

def main():
    args = init_args()
    logger.info(f"cuda is available:{torch.cuda.is_available()} ")
    wandb_logger = init_wandb()
    run_name = wandb_logger.experiment.name
    logger.info(run_name)
    repo_path = init_repo(run_name)
    module = load_module(
        train_bs = args.train_bs,
        test_bs = args.test_bs
    )

    model = load_model()
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        default_root_dir= repo_path,
        accelerator="auto", 
        devices="auto", 
        strategy="auto"
    )
    trainer.fit(model, module)


if __name__ == "__main__":
    main()