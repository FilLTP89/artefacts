import os
import torch
import random 
import numpy as np
import pytorch_lightning as pl  
from data_file.processing_newdata import Datav2Module, load_one_acquisition, LoadOneAcquisition
from model.torch.Attention_MEDGAN import AttentionMEDGAN
import pytorch_lightning as pl
import matplotlib.pyplot as plt

CPKT_PATH = "/lustre/fswork/projects/rech/xvy/upz57sx/artefact/model/saved_model/best_model-epoch=10-test_mse_loss=0.23.ckpt"


def save_image(
        x, 
        preds,
        y, 
        path, 
        idx
    ):
    x = x.squeeze(0)
    preds = preds.squeeze(0)
    y = y.squeeze(0)

    cmap = plt.cm.gray
    x = x.cpu().numpy()
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()

    mse = np.mean((preds - y)**2)
    print(f"MSE: {mse}")
    
    plt.imsave(
        path + f"{idx}_original_image" + ".png",
        x,
        cmap=cmap,
    )
    plt.imsave(
        path  + f"{idx}_predicted_image" + ".png",
        preds,
        cmap=cmap,
    )
    plt.imsave(
        path + f"{idx}_ground_truth_image" + ".png",
        y,
        cmap=cmap,
    )

    

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
                    dataloader, 
                    saving_path,
                    device = "cpu",
                    run_name = "run_name",
                    ):
    print(f"Saving in folder {saving_path + run_name}")
    for idx_, batch in enumerate(dataloader):
        x,y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            generated = model(x)
        for i in range(generated.size(0)):
            save_image(
                x[i], generated[i], y[i], 
                path = saving_path + run_name,
                idx = idx_*generated.size(0) + i
            )

        
        
def main():
    i = 0
    acquisition_number = 1
    categorie = "huttalowmetal"
    folder_path = f"datav2/protocole_1/control/{acquisition_number}/dcm/Input/{categorie}/"

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
    )
    print("Dataset size : ", len(ds))
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size = 16,
        shuffle = False
    ) 
    print("Model loaded")
    generate_images(model = model, 
                    dataloader = dataloader, 
                    saving_path = saving_path,
                    run_name = run_name,
                    device = device
                    )
    print("Images generated")

if __name__ == "__main__":
    main()