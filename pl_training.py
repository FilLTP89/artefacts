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
from pytorch_lightning.utilities import rank_zero_only
import logging
import tempfile
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

work_dir = "/gpfs/users/gabrielihu/tmp"  # Replace this with your actual work directory path
tempfile.tempdir = work_dir

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_float32_matmul_precision('medium')

SAVE_WEIGHTS_ONLY = False
VGG_CPKT = "model/saved_model/best_model-epoch=19-val_acc=0.94.ckpt"
ATTENTION_MEDGAN_CPKT = "model/saved_model/best_model-epoch=19-test_mse_loss=0.00.ckpt"

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("CustomModelCheckpoint initialized")
        
    def _save_checkpoint(self, trainer, filepath):
        print("\n\n----- CUSTOM CHECKPOINT INFO -----")
        print(f"Saving model to: {filepath}")
        model_size = self._get_model_size(trainer.model)
        num_files = self._count_checkpoint_files(filepath)
        print(f"Model size: {model_size:.2f} MB")
        print(f"Number of files to be saved: {num_files}")
        print("----- END CUSTOM CHECKPOINT INFO -----\n\n")
        super()._save_checkpoint(trainer, filepath)

    def _get_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def _count_checkpoint_files(self, filepath):
        # PyTorch Lightning saves the checkpoint as a single file
        return 1

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
    parser.add_argument("--data_folder", type=str, default="complete")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    args = parser.parse_args()
    return args



def init_wandb():
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
            project=f"Metal_Artifacts_Reduction",
    )
    return wandb_logger


def init_repo(wandb_name,
              model_name = None, 
              ruche = False):
    if not ruche:
        path = f"model/saved_model/{model_name}/{wandb_name}"
        os.makedirs(path, exist_ok=True)
    else:
        path = f"/gpfs/workdir/gabrielihu/artefacts/model/saved_model/{model_name}/{wandb_name}"
        os.makedirs(path, exist_ok=True)
    return path


def load_module(
        task = "GAN",
        data_folder = None,
        train_bs = 1,
        test_bs = 1,
        *args, **kwargs):
    if (task == "GAN") or (task == "Diffusion"):
        module = Datav2Module(dataset_type = Datav2Dataset,
                              data_folder = data_folder, 
                              train_bs = train_bs,
                              test_bs = test_bs,
                              *args, 
                              **kwargs)
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
    module = load_module(
        train_bs = args.train_bs,
        test_bs = args.test_bs,
        task= args.task,
        data_folder = args.data_folder
    )
    if args.use_feature_extractor:
        feature_extractor = load_feature_extractor()
    else :
        feature_extractor = None
    model = load_model(
        learning_rate = args.lr,
        task= args.task,
        n_class = module.n_class,
        feature_extractor = feature_extractor,
        resume_from_cpkt = args.resume_from_cpkt    
    )
    model_name = type(model).__name__
    #model = torch.compile(model)
    
    repo_path = init_repo(wandb_name = run_name, model_name = model_name, ruche = args.ruche)
    
    rank_zero_info("\n \n \n ")
    rank_zero_info(f"Task : {args.task}")
    rank_zero_info(f"Cuda is available:{torch.cuda.is_available()}, gpu available: {device_count}")
    rank_zero_info(f"Run name {run_name}")
    rank_zero_info(f"Run path {repo_path}")
    rank_zero_info(f"Use mix precision : {args.mix_precision}")
    rank_zero_info(f"Use one batch : {args.one_batch}")
    rank_zero_info(f"Save weight only : {SAVE_WEIGHTS_ONLY}")
    rank_zero_info(f"Data used : {args.data_folder}")


    if args.resume_from_cpkt:
        rank_zero_info(f"Resume from checkpoint : {args.resume_from_cpkt}, checkpoint path : {ATTENTION_MEDGAN_CPKT}")
    
    rank_zero_info(f"Model name : {model_name}")
    rank_zero_info("\n \n \n ")
    monitor_dict = {
        "AttentionMEDGAN": ("test_mse_loss","min"),
        "VGG19": ("val_acc","max"),
        "Diffusion_UNET": ("MSE_loss","min")
    }
    callbacks = [
            CustomModelCheckpoint(
        dirpath = repo_path,
        filename="best_model-{epoch:02d}-{" + monitor_dict[model_name][0] + ":.2f}",
        save_top_k = 1,
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
        strategy="ddp_find_unused_parameters_true" if model_name == "AttentionMEDGAN" else "ddp",
        overfit_batches= 1 if args.one_batch else 0,
        num_nodes=1,
        callbacks=callbacks,
        precision="bf16-mixed" if args.mix_precision else 32,
        log_every_n_steps=1,
        accumulate_grad_batches= args.accumulate_grad_batches,
    )
    trainer.fit(model, 
                train_dataloaders = module.train_dataloader(),
                val_dataloaders = module.val_dataloader(),
                ckpt_path= None if not args.resume_from_cpkt else ATTENTION_MEDGAN_CPKT
                )


if __name__ == "__main__":
    main()