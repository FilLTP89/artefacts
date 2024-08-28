import os
import torch
import random 
import numpy as np
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module
from model.torch.Attention_MEDGAN import AttentionMEDGAN
import pytorch_lightning as pl


RUN_PATH = ""
CPKT_NAME = "model/saved_models/torch/"




def load_model(checkpoint_path, *args, **kwargs):
    model = AttentionMEDGAN(*args, **kwargs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location = device))
    model.eval()
    return model


def load_module(*args, **kwargs):
    module = Datav2Module(*args, **kwargs)
    module.setup()
    return module


def main(
        saving_patch = "generated_images/",):
    model = load_model("model/saved_models/torch/")
    module = load_module()
