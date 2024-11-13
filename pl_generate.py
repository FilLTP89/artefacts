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

CPKT_PATH = "model/saved_model/AttentionMEDGAN/best_model/best_model-epoch=19-test_mse_loss=0.00.ckpt"



    

def load_model(checkpoint_path, device, *args, **kwargs):
    model = AttentionMEDGAN.load_from_checkpoint(checkpoint_path=checkpoint_path, filters =  [8,16,32, 64,128,256,512,1024])
    model = model.to(device)
    model.eval()
    return model


def load_module(*args, **kwargs):
    module = Datav2Module(*args, **kwargs)
    module.setup()
    return module

def generate_images(model, 
                    ds, 
                    saving_path,
                    device = "cpu",
                    run_name = "run_name",
                    ):
    print(f"Saving in folder {saving_path + run_name}")
    for idx_, (input,target,input_path,target_path) in enumerate(ds):
        input = input.to(device)
        with torch.no_grad():
            print(input.unsqueeze(0).shape)
            generated = model(input.unsqueeze(0))   
            generated = generated.squeeze(0) 
            generated = generated.cpu().detach().numpy()
            input = input.cpu().detach().numpy()

            input_dcm = dicom.dcmread(input_path)
            target_dcm = dicom.dcmread(target_path)  
            bit_depth = input_dcm.BitsStored if hasattr(input_dcm, 'BitsStored') else 16
            max_val = float(2**bit_depth - 1)
            input = input * max_val
            target = target * max_val
            generated = generated * max_val

            if hasattr(input_dcm, 'RescaleSlope') and hasattr(input_dcm, 'RescaleIntercept'):
                """
                Actually input and output have the same RescaleSlope and RescaleIntercept
                """
                input_array = (input_array - input_dcm.RescaleIntercept) / input_dcm.RescaleSlope
                generated_array = (generated_array - input_dcm.RescaleIntercept) / input_dcm.RescaleSlope
                target_array = (target_array - target_dcm.RescaleIntercept) / target_dcm.RescaleSlope
            
            input_array = input_array.astype(input_dcm.pixel_array.dtype)
            target_array = target_array.astype(target_dcm.pixel_array.dtype)
            generated_array = generated_array.astype(target_dcm.pixel_array.dtype)

            new_input_dcm = input_array.astype(input_dcm.pixel_array.dtype)
            new_target_dcm = target_array.astype(target_dcm.pixel_array.dtype)
            new_generated_dcm = generated_array.astype(target_dcm.pixel_array.dtype) 


            new_input_dcm.PixelData = input_array.tobytes()
            new_target_dcm.PixelData = target_array.tobytes()
            new_generated_dcm.PixelData = generated_array.tobytes()

            # Generate new UIDs
            new_input_dcm.SOPInstanceUID = dicom.uid.generate_uid()
            new_target_dcm.SOPInstanceUID = dicom.uid.generate_uid()
            new_generated_dcm.SOPInstanceUID = dicom.uid.generate_uid()

            new_input_dcm.save_as(saving_path + run_name + f"input_{idx_}.dcm")
            new_target_dcm.save_as(saving_path + run_name + f"target_{idx_}.dcm")
            new_generated_dcm.save_as(saving_path + run_name + f"generated_{idx_}.dcm")


        
def main():
    i = 0
    acquisition_number = 4
    categorie = "controllowmetal"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saving_path = "new_generated/"
    run_name = f"test_{i}/"
    while os.path.exists(saving_path + run_name):
        i = i+1
        run_name = f"test_{i}/"
    os.makedirs(saving_path + run_name)
    
    model = load_model(
        checkpoint_path=CPKT_PATH,
        device = device,
    )
    model = model.to(device)


    # Count only the files (not directories)
    ds = LoadOneAcquisition(
        categorie = categorie,
        acquisition = acquisition_number,
        generating=True
    )
    print("Dataset size : ", len(ds))
    print("Model loaded")
    generate_images(model = model, 
                    ds = ds, 
                    saving_path = saving_path,
                    run_name = run_name,
                    device = device
                    )
    print("Images generated")

if __name__ == "__main__":
    main()