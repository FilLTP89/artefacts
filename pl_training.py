import os
import torch
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRAIN_BS = 16
TEST_BS = 16



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



def main():
    logger.info(f"cuda is available:{torch.cuda.is_available()} ")
    wandb_logger = init_wandb()
    run_name = wandb_logger.experiment.name
    print(run_name)
    repo_path = init_repo(run_name)
    module = load_module()


if __name__ == "__main__":
    main()