import os
import torch
import random 
import numpy as np
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module, Datav2Dataset, ClassificationDataset
from model.torch.Attention_MEDGAN import AttentionMEDGAN, VGG19
from model.torch.DIffusion_UNET import Diffusion_UNET
import pytorch_lightning as pl
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.set_float32_matmul_precision("medium")

SAVE_WEIGHTS_ONLY = False
VGG_CPKT = "model/saved_model/best_model-epoch=19-val_acc=0.94.ckpt"
ATTENTION_MEDGAN_CPKT = "model/saved_model/best_model-epoch=19-test_mse_loss=0.00.ckpt"


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
    parser.add_argument("--use_feature_extractor", action=argparse.BooleanOptionalAction,type=bool, default=True)   
    parser.add_argument("--one_batch",action=argparse.BooleanOptionalAction, type=bool, default=False)    
    parser.add_argument("--mix_precision", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--ruche", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--task", type=str, default="GAN")
    parser.add_argument("--resume_from_cpkt", action = argparse.BooleanOptionalAction, type=bool, default=False)
    args = parser.parse_args()
    return args



def init_wandb():
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
            project=f"Metal_Artifacts_Reduction",
    )
    return wandb_logger


def init_repo(wandb_name, ruche = False):
    if not ruche:
        path = f"model/saved_model/{wandb_name}"
        os.makedirs(path, exist_ok=True)
    else:
        path = f"model/saved_model/{wandb_name}/"
        os.makedirs(path, exist_ok=True)
    return path


def load_module(
        task = "GAN",
        *args, **kwargs):
    if (task == "GAN") or (task == "Diffusion"):
        module = Datav2Module(dataset_type = Datav2Dataset,*args, **kwargs)
    else:
        module = Datav2Module(dataset_type = ClassificationDataset,*args, **kwargs)
    module.setup()
    return module

def load_model(task ="GAN",
               n_class = None,
               resume_from_cpkt = False,
               *args, **kwargs):
    if task == "GAN":
        if resume_from_cpkt:
            vgg = VGG19(classifier_training= False, n_class=15, load_whole_architecture=True)
            model = AttentionMEDGAN(feature_extractor = vgg)
        else: 
            model = AttentionMEDGAN(*args, **kwargs)
    elif task == "Diffusion":
        model = Diffusion_UNET(in_channels=1)
    else:
        model = VGG19(classifier_training= True,
                      n_class=n_class, 
                      *args, **kwargs)
    return model

def load_feature_extractor(*args, **kwargs):
    model = VGG19.load_from_checkpoint(VGG_CPKT, 
                                       load_whole_architecture = True,
                                       classifier_training=False,
                                       n_class = 15)
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
        test_bs = args.test_bs,
        task= args.task,
    )
    if args.use_feature_extractor:
        feature_extractor = load_feature_extractor()
    else :
        feature_extractor = None

    
    rank_zero_info("\n \n \n ")
    rank_zero_info(f"Task : {args.task}")
    rank_zero_info(f"Cuda is available:{torch.cuda.is_available()}, gpu available: {device_count}")
    rank_zero_info(f"Run name {run_name}")
    rank_zero_info(f"Run path {repo_path}")
    rank_zero_info(f"Use mix precision : {args.mix_precision}")
    rank_zero_info(f"Use one batch : {args.one_batch}")
    rank_zero_info(f"Save weight only : {SAVE_WEIGHTS_ONLY}")
    if args.resume_from_cpkt:
        rank_zero_info(f"Resume from checkpoint : {args.resume_from_cpkt}, checkpoint path : {ATTENTION_MEDGAN_CPKT}")
    rank_zero_info("\n \n \n ")
    
    model = load_model(
        learning_rate = args.lr,
        task= args.task,
        n_class = module.n_class,
        feature_extractor = feature_extractor,
        resume_from_cpkt = args.resume_from_cpkt    
    )
    
    model_name = type(model).__name__
    saving_path = f"{repo_path}/{model_name}"
    os.makedirs(saving_path , exist_ok=True)
    monitor_dict = {
        "AttentionMEDGAN": ("test_mse_loss","min"),
        "VGG19": ("val_acc","max"),
        "Diffusion_UNET": ("MSE_loss","min")
    }
    callbacks = [
            ModelCheckpoint(
        dirpath = saving_path,
        filename = "best_model-{epoch:02d}-{test_mse_loss:.2f}" if model_name == "AttentionMEDGAN" else "best_model-{epoch:02d}-{val_acc:.2f}",
        save_top_k =1,
        verbose = True,   
        monitor = monitor_dict[model_name][0],
        mode = monitor_dict[model_name][1], 
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
                val_dataloaders = module.val_dataloader(),
                ckpt_path= None if not args.resume_from_cpkt else ATTENTION_MEDGAN_CPKT
                )


if __name__ == "__main__":
    main()