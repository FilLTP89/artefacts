import os
import torch
import random 
import numpy as np
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module, load_one_acquisition, LoadOneAcquisition
from model.torch.Attention_MEDGAN import AttentionMEDGAN
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pydicom as dicom



CPKT_PATH = "model/saved_model/AttentionMEDGAN/zesty-monkey-164/best_model-epoch=03-test_mse_loss=0.00.ckpt"


def load_model(checkpoint_path, device, *args, **kwargs):
    model = AttentionMEDGAN.load_from_checkpoint(checkpoint_path=checkpoint_path, filters =  [8,16,32, 64,128,256,512,1024])
    model = model.to(device)
    model.eval()
    return model


def main():
    print("Start")
    i = 0
    acquisition_number = 1
    dcm = "dcm"
    control = "fracture"
    categories = os.listdir(f"datav2/protocole_1/{control}/{acquisition_number}/{dcm}/Input/")
    print(f"Categorie : {categories}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(
        checkpoint_path=CPKT_PATH,
        device = device,
    ).to(device)
    model.eval()
    print("Model loaded")
    for categorie in categories:
        saving_path = f"new_generated/complete/"
        run_name = f"{categorie}/"
        test_name = f"test_0/"
        while os.path.exists(saving_path + str(acquisition_number) + "/" + run_name + test_name):
            i = i+1
            test_name = f"test_{i}/" 
        saving_path = saving_path + str(acquisition_number) + "/" + run_name + test_name
        os.makedirs(saving_path)
        print(f"Directories {saving_path} created")

        # Count only the files (not directories)
        ds = LoadOneAcquisition(
            categorie = categorie,
            acquisition = acquisition_number,
            generating=True,
            control=True if control == "control" else False,
        )
        print("Dataset size : ", len(ds))
        print(f"Saving path : {saving_path}")
        ds.generate(model = model, 
                    output_dir = saving_path,
                    device = device
                    )
        print("Images generated")

if __name__ == "__main__":
    main()